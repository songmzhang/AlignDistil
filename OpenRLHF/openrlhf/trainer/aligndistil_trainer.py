import itertools
import math
import os
import socket
import time
import numpy as np
from abc import ABC
from datetime import timedelta
from typing import Callable, Dict, List

import deepspeed
import ray
import torch
from torch import nn
import torch.distributed as dist
from torch.optim import Optimizer
from tqdm import tqdm
from transformers.trainer import get_scheduler
from deepspeed.utils import logger

from openrlhf.datasets import SFTDataset
from openrlhf.datasets.utils import zero_pad_sequences
from openrlhf.models import GPTLMLoss
from openrlhf.trainer.ppo_utils import RemoteKDExperienceMaker
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.utils.distributed_util import init_process_group


class AlignDistilTrainer(ABC):
    def __init__(
        self,
        model,
        teacher_model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm: float = 1,
        pretrain_mode: bool = False,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
        ref_model: ray.actor.ActorHandle = None,
        vllm_engines: List = None,
        **generation_kwargs,

    ):
        """
        Trainer for On-Policy AlignDistill.

        Args:
            model (torch.nn.Module): The model to be trained.
            teacher_model (ray.actor.ActorHandle): The forward DPO model.
            strategy (Strategy): The training strategy to be applied.
            optim (Optimizer): The optimizer for model training.
            train_dataloader (DataLoader): The dataloader for the training dataset.
            eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
            scheduler (Scheduler): The learning rate scheduler to adjust training rates.
            max_norm (float, defaults to 1): Maximum gradient norm for clipping to prevent exploding gradients.
            pretrain_mode (bool, defaults to False): Flag to indicate if the trainer is in pre-training mode.
            batch_size (int, defaults to 1): Batch size for training.
            max_epochs (int, defaults to 2): The maximum number of training epochs.
            tokenizer (Tokenizer, optional): The tokenizer for processing input data.
            ref_model (ray.actor.ActorHandle, defaults to None): The reverse DPO model.
            vllm_engines (List, defaults to None): vllm engines for rollout.
        """
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args
        self.ref_model = ref_model
        self.vllm_engines = vllm_engines
        
        self.loss_fn = GPTLMLoss()

        self.experience_maker = RemoteKDExperienceMaker(
            self.model,
            None,
            None,
            None,
            self.tokenizer,
            self.args.prompt_max_len,
            None,
            self.strategy,
            None,
            None,
            vllm_engines=self.vllm_engines,
            packing_samples=self.args.packing_samples,
            teacher_model=self.teacher_model,
            ref_model=self.ref_model
        )

        # Create torch group with deepspeed rank 0 and all vllm ranks
        # to update vllm engine's weights after each training stage.
        #
        # Say we have 3 vllm engines and eache of them has 4 GPUs,
        # then the torch group is:
        # [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
        # |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        if self.vllm_engines is not None and torch.distributed.get_rank() == 0:
            # master_address = ray._private.services.get_node_ip_address()
            # with socket.socket() as sock:
            #     sock.bind(("", 0))
            #     master_port = sock.getsockname()[1]
            master_address = os.environ.get("MASTER_ADDR", ray._private.services.get_node_ip_address())
            # master_address = "dlc1gbif47ch01av-master-0"
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = int(os.environ.get("MASTER_PORT", sock.getsockname()[1]))

            vllm_num_engines, vllm_tensor_parallel_size = (
                self.strategy.args.vllm_num_engines,
                self.strategy.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

            backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
            # https://github.com/OpenRLHF/OpenRLHF/issues/313
            import vllm

            if not vllm.__version__ == "0.4.2" and not vllm.__version__ >= "0.6.4":
                backend = "gloo"
                print(
                    "Warning: using --vllm_sync_backend=gloo for `not vLLM version == 0.4.2 and not vllm.__version__ >= 0.6.4`"
                )

            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    "openrlhf",
                    backend=backend,
                )
                for i, engine in enumerate(self.vllm_engines)
            ]
            self._model_update_group = init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name="openrlhf",
            )

            ray.get(refs)

        torch.distributed.barrier()
            
    def _broadcast_to_vllm(self):
        # avoid OOM
        torch.cuda.empty_cache()
        model = self.model.model.module
        count, num_params = 0, len(list(model.named_parameters()))
        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param
            
            if "value_head" in name:
                continue

            # Fire all vllm engines for broadcast
            if torch.distributed.get_rank() == 0:
                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight.remote(name, dtype=param.dtype, shape=shape, empty_cache=count == num_params)
                    for engine in self.vllm_engines
                ]

            # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
            with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                if torch.distributed.get_rank() == 0:
                    torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
                    ray.get(refs)


    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        self.strategy.accumulated_gradient = self.strategy.accumulated_gradient // args.n_samples_per_prompt
        num_update_steps_per_epoch = num_update_steps_per_epoch * args.n_samples_per_prompt
        global_logs_dict = {}

        if self.strategy.is_rank_0():
            logger.info("++++++++++++++ Start Training ++++++++++++++")
        
        start_time = time.time()
        init_reward = -100
        global_step = 0
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )
                
            # train
            self.model.train()
            
            for rand_prompts in self.train_dataloader:
                # collect a global batch of prompt and generate together to speed up
                experience_list = self.experience_maker.make_experience_list(rand_prompts, **self.generation_kwargs)
                
                self.model.train()
                for experience_batch in experience_list:
                    sequences = experience_batch.sequences
                    teacher_logits = experience_batch.teacher_logits
                    reference_logits = experience_batch.reference_logits
                    attention_mask = experience_batch.attention_mask
                    action_mask = experience_batch.action_mask
                    attention_mask = experience_batch.attention_mask
                    prompt_len = experience_batch.sequences.size(-1) - action_mask.size(-1)
                    response_length = experience_batch.info["response_length"]
                    total_length = experience_batch.info["total_length"]
                    
                    # teacher_logits = teacher_logits.to(torch.cuda.current_device())
                    # reference_logits = reference_logits.to(torch.cuda.current_device())
                    
                    labels = torch.where(
                        attention_mask.bool(),
                        torch.cat([sequences[:, 1:], torch.ones_like(sequences)[:, :1] * (-100)], -1),
                        self.loss_fn.IGNORE_INDEX,
                    )
                    for label in labels:
                        label[:prompt_len-1] = self.loss_fn.IGNORE_INDEX
                    
                    output = self.model(sequences, attention_mask=attention_mask, return_output=True)
                    student_logits = output.logits
                    # discard the prompt part to save GPU memory (prompts did left padding and end at the same position)
                    student_logits = student_logits[:, prompt_len:, :]
                    labels = labels[:, prompt_len:]
                    
                    mask = (labels != -100).int()
                    label_ids = torch.where(labels.eq(-100), self.tokenizer.pad_token_id, labels)
                
                    # prune logits with different sizes (only for qwen)
                    if student_logits.size(-1) != teacher_logits.size(-1):
                        min_logit_size = min(student_logits.size(-1), teacher_logits.size(-1))
                        student_logits = student_logits[:, :, :min_logit_size]
                        teacher_logits = teacher_logits[:, :, :min_logit_size]

                    stu_probs = torch.softmax(student_logits / args.kd_temperature, dim=-1, dtype=torch.float32)
                    stu_lprobs = torch.log_softmax(student_logits / args.kd_temperature, dim=-1, dtype=torch.float32)
                    tea_lprobs = torch.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
                    ref_lprobs = torch.log_softmax(reference_logits, dim=-1, dtype=torch.float32)
                        
                    reward_distribution = args.beta * (tea_lprobs - ref_lprobs)
                    all_reward = reward_distribution.gather(-1, label_ids.unsqueeze(-1)).squeeze(-1)
                    expected_reward = ((stu_probs * reward_distribution).sum(-1) * mask).sum(-1).mean(0)
                    dpo_model_expected_reward = ((tea_lprobs.exp() * reward_distribution).sum(-1) * mask).sum(-1).mean(0)
                    
                    if args.reward_boost_type == "theorem1":
                        weight = args.beta / args.beta2
                        final_tea_logits = weight * teacher_logits + (1 - weight) * reference_logits
                        final_tea_lprobs = torch.log_softmax(final_tea_logits / args.kd_temperature, dim=-1, dtype=torch.float32)
                        rlhf_loss = (stu_probs * (stu_lprobs - final_tea_lprobs)).sum(-1) * args.beta2
                    elif args.reward_boost_type == "theorem1_contrast":
                        weight = args.beta / args.beta2
                        final_tea_logits = weight * (teacher_logits - reference_logits) + teacher_logits
                        final_tea_lprobs = torch.log_softmax(final_tea_logits / args.kd_temperature, dim=-1, dtype=torch.float32)
                        rlhf_loss = (stu_probs * (stu_lprobs - final_tea_lprobs)).sum(-1) * args.beta2
                    elif args.reward_boost_type == "theorem1_adaptive":
                        tvd = (tea_lprobs.exp() - ref_lprobs.exp()).abs().sum(-1)
                        weight = (tvd * args.beta2 + 1e-3)
                        beta2 = args.beta / weight
                        final_tea_logits = weight.unsqueeze(-1) * teacher_logits + (1 - weight.unsqueeze(-1)) * reference_logits
                        final_tea_lprobs = torch.log_softmax(final_tea_logits / args.kd_temperature, dim=-1, dtype=torch.float32)
                        rlhf_loss = (stu_probs * (stu_lprobs - final_tea_lprobs)).sum(-1) * beta2
                    elif args.reward_boost_type == "aligndistil":
                        tvd = (tea_lprobs.exp() - ref_lprobs.exp()).abs().sum(-1)
                        weight = (tvd * args.beta2 + 1e-3)
                        beta2 = args.beta / weight
                        final_tea_logits = weight.unsqueeze(-1) * (teacher_logits - reference_logits) + teacher_logits
                        final_tea_lprobs = torch.log_softmax(final_tea_logits / args.kd_temperature, dim=-1, dtype=torch.float32)
                        rlhf_loss = (stu_probs * (stu_lprobs - final_tea_lprobs)).sum(-1) * beta2
                    elif args.reward_boost_type == "gkd":   # for GKD with RKL
                        rlhf_loss = (stu_probs * (stu_lprobs - tea_lprobs)).sum(-1)

                    seq_reward = (all_reward * mask).sum(-1).mean(0)
                    token_reward = ((all_reward * mask).sum(-1) / mask.sum(-1)).mean(0)
                    rkl_with_tea = (stu_probs * (stu_lprobs - tea_lprobs)).sum(-1)
                    rkl_with_ref = (stu_probs * (stu_lprobs - ref_lprobs)).sum(-1)
                    seq_init_kl = (rkl_with_ref * mask).sum(-1).mean(0)
                    seq_tea_kl = (rkl_with_tea * mask).sum(-1).mean(0)
                    token_init_kl = ((rkl_with_ref * mask).sum(-1) / mask.sum(-1)).mean(0)
                    token_tea_kl = ((rkl_with_tea * mask).sum(-1) / mask.sum(-1)).mean(0)
                    
                    if not isinstance(weight, torch.Tensor):
                        weight = torch.tensor(weight)
                    if weight.dim() > 1:
                        weight = ((weight * mask).sum(-1) / mask.sum(-1)).mean(0)
                    
                    if init_reward == -100:
                        init_reward = expected_reward
                    
                    rlhf_loss = (rlhf_loss * mask).sum(-1)

                    if args.reward_len_norm:
                        rlhf_loss = rlhf_loss / mask.sum(-1)

                    rlhf_loss = rlhf_loss.mean(0)
                    
                    gpt_loss = torch.zeros(1).to(torch.cuda.current_device())
                    loss = gpt_loss * (1 - self.args.kd_coef) + rlhf_loss * self.args.kd_coef
                    self.strategy.backward(loss, self.model, self.optimizer)
                    self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
            
                    # We don't need moving average if we log the averaged results every args.logging_steps
                    # loss_mean = loss_mean * 0.9 + 0.1 * gpt_loss.item()
                    logs_dict = {
                        "gpt_loss": gpt_loss.item(),
                        "rlhf_loss": rlhf_loss.item(),
                        "mean_weight": weight.item(),
                        "reward/kl": (expected_reward.item() - init_reward.item()) / (seq_init_kl.item() + 1),
                        "expected_reward": expected_reward.item(),
                        "dpo_model_expected_reward": dpo_model_expected_reward.item(),
                        "seq_reward": seq_reward.item(),
                        "token_reward": token_reward.item(),
                        "seq_init_kl": seq_init_kl.item(),
                        "token_init_kl": token_init_kl.item(),
                        "seq_tea_kl": seq_tea_kl.item(),
                        "token_tea_kl": token_tea_kl.item(),
                        "response_length": sum(response_length) / len(response_length),
                        "total_length": sum(total_length) / len(total_length),
                        "lr": self.scheduler.get_last_lr()[0],
                    }
                    # step bar
                    logs_dict = self.strategy.all_reduce(logs_dict)
                    # step_bar.set_postfix(logs_dict)
                    # step_bar.update()
                    
                    # add the log in per micro_step into the global log
                    for k in logs_dict:
                        if k not in global_logs_dict:
                            global_logs_dict[k] = [logs_dict[k]]
                        else:
                            global_logs_dict[k].append(logs_dict[k])  
                    
                    step += 1            

                # logs/checkpoints/evaluation
                global_step = step // self.strategy.accumulated_gradient
                # client_states = {"consumed_samples": global_step * args.train_batch_size}
                # self.save_logs_and_checkpoints(args, global_step, None, logs_dict, client_states)
                
                vllm_broadcast_time = 0
                # broadcast student parameter to vllm
                if self.vllm_engines is not None and global_step % args.vllm_sync_steps == 0:
                    torch.distributed.barrier()
                    self._broadcast_to_vllm()
                    
                if global_step % args.logging_steps == 0 or step == len(self.train_dataloader) - 1:
                    for k in global_logs_dict:
                        if isinstance(global_logs_dict[k], list) and len(global_logs_dict[k]) > 0:
                            global_logs_dict[k] = sum(global_logs_dict[k]) / len(global_logs_dict[k])
                    
                    progress = global_step / num_update_steps_per_epoch / self.epochs
                    eta = int(time.time() - start_time) * (1 - progress) / progress
                    progress_str = "epoch [{current_epoch}/{total_epoch}], " \
                        "step [{current_step}/{total_step}], " \
                        "train_progress [{progress:.2f}%], " \
                        "Elapsed: {elapsed}, " \
                        "ETA: {eta}, ".format(
                        current_epoch=epoch, 
                        total_epoch=self.epochs, 
                        current_step=global_step, 
                        total_step=num_update_steps_per_epoch * self.epochs, 
                        progress=progress * 100,
                        elapsed=str(timedelta(seconds=(time.time() - start_time))).split(".")[0],
                        eta=str(timedelta(seconds=eta)).split(".")[0]
                    )
                    if self.strategy.is_rank_0():
                        log_info = []
                        for k in global_logs_dict:
                            if k == "lr":
                                log_info.append(f"lr: {global_logs_dict[k]:.6e}")
                            else:
                                log_info.append(f"{k}: {global_logs_dict[k]:.6f}")
                        log_str = ", ".join(log_info)
                        log_str = progress_str + log_str
                        logger.info(log_str)
                    for k in global_logs_dict:
                        global_logs_dict[k] = []
                        
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.save_path, f"step_{global_step}")
                    self.strategy.save_model(self.model.model, self.tokenizer, save_path)
                

            # epoch_bar.update()
            # save checkpoint after every epoch
            # tag = f"epoch{epoch+1}"
            # client_states = {"consumed_samples": 0}
            # self.strategy.save_ckpt(
            #     self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
            # )
            
        total_time = time.time() - start_time
        if self.strategy.is_rank_0():
            logger.info(f"Training done, totally cost {str(timedelta(seconds=total_time)).split('.')[0]}")

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if global_step % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(
                self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
            )

    def evaluate(self, eval_dataloader, steps=0):
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            for prompts_id_len, inputs, attention_masks, _ in eval_dataloader:
                inputs = inputs.squeeze(1).to(torch.cuda.current_device())
                attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
                logits = self.model(inputs, attention_mask=attention_mask, return_output=True)["logits"]

                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )
                if not self.pretrain_mode:
                    for label, source_len in zip(labels, prompts_id_len):
                        label[:source_len] = self.loss_fn.IGNORE_INDEX
                loss = self.loss_fn(logits, labels)

                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval gpt_loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)

        self.model.train()  # reset model state
