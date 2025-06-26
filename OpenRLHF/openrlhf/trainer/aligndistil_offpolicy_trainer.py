import math
import os
import time
import numpy as np
from datetime import timedelta
from abc import ABC

import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm
from transformers.trainer import get_scheduler
from deepspeed.utils import logger

from openrlhf.datasets import SFTDataset
from openrlhf.datasets.utils import zero_pad_sequences
from openrlhf.models import GPTLMLoss, KDLoss
from openrlhf.utils.distributed_sampler import DistributedSampler


class AlignDistilOffPolicyTrainer(ABC):
    """
    Trainer for Off-Policy RL with DPO Reward.

    Args:
        model (torch.nn.Module): The model to be trained.
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
    """

    def __init__(
        self,
        model,
        teacher_model,
        ref_model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm: float = 1,
        pretrain_mode: bool = False,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None
    ) -> None:
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
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args

        self.loss_fn = GPTLMLoss()
        self.kd_loss = KDLoss(
            divergence=strategy.args.kd_divergence, 
            temperature=strategy.args.kd_temperature
        )
        
        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

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
            self.teacher_model.eval()
            self.ref_model.eval()
            
            for data in self.train_dataloader:
                # collect a global batch of prompt and generate together to speed up
                chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                
                # input_ids, att_masks, prompt_id_lens = self.concatenated_inputs(
                #     chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                # )
                input_ids = chosen_ids
                att_masks = c_mask
                
                labels = torch.where(
                    att_masks.bool(),
                    input_ids,
                    self.loss_fn.IGNORE_INDEX,
                )

                if not self.pretrain_mode:
                    for label, source_len in zip(labels, prompt_id_lens):
                        label[:source_len] = self.loss_fn.IGNORE_INDEX
                        
                student_logits = self.model(input_ids, attention_mask=att_masks, return_output=True)["logits"]
                with torch.no_grad():
                    teacher_logits = self.teacher_model(
                        input_ids, attention_mask=att_masks, return_output=True
                    )["logits"]
                    reference_logits = self.ref_model(
                        input_ids, attention_mask=att_masks, return_output=True
                    )["logits"]
                
                # torch.cuda.empty_cache()
                
                # prune logits with different sizes (only for qwen)
                if student_logits.size(-1) != teacher_logits.size(-1):
                    min_logit_size = min(student_logits.size(-1), teacher_logits.size(-1))
                    student_logits = student_logits[:, :, :min_logit_size]
                    teacher_logits = teacher_logits[:, :, :min_logit_size]
                
                # if input_ids.size(0) == 2:
                #     student_logits = student_logits[:, prompt_id_lens[0]:, :]
                #     teacher_logits = teacher_logits[:, prompt_id_lens[0]:, :]
                #     reference_logits = reference_logits[:, prompt_id_lens[0]:, :]
                #     labels = labels[:, prompt_id_lens[0]:]
                
                mask = (labels != -100).int()
                label_ids = torch.where(labels.eq(-100), self.tokenizer.pad_token_id, labels)
                
                stu_probs = torch.softmax(student_logits, dim=-1, dtype=torch.float32)
                stu_lprobs = torch.log_softmax(student_logits, dim=-1, dtype=torch.float32)
                tea_lprobs = torch.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
                ref_lprobs = torch.log_softmax(reference_logits, dim=-1, dtype=torch.float32)
                    
                reward_distribution = args.beta * (tea_lprobs - ref_lprobs)
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
                    "seq_init_kl": seq_init_kl.item(),
                    "token_init_kl": token_init_kl.item(),
                    "seq_tea_kl": seq_tea_kl.item(),
                    "token_tea_kl": token_tea_kl.item(),
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

                # train_time = time.time() - train_start
                # logs/checkpoints/evaluation
                global_step = step // self.strategy.accumulated_gradient
                # client_states = {"consumed_samples": global_step * args.train_batch_size}
                # self.save_logs_and_checkpoints(args, global_step, None, logs_dict, client_states)
                
                if step % self.strategy.accumulated_gradient == 0:
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

                # global_batch_time = time.time() - vllm_start
                # if self.strategy.is_rank_0():
                #     print(f"global_batch_time: {global_batch_time}, vllm_time: {vllm_time}, prepare_time: {prepare_time}, train_time: {train_time}, vllm_broadcast_time: {vllm_broadcast_time}")
                
                

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

            # epoch_bar.update()

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
        
    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks, prompt_id_lens = self.concatenated_inputs(
            chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
        )
        output = model(input_ids, attention_mask=att_masks, return_output=True)
        all_logits = output["logits"]
        all_logps_sum, all_logps_mean = self._get_batch_logps(
            all_logits, input_ids, att_masks, prompt_id_lens, average_log_prob=False
        )
        chosen_logps = all_logps_sum[: chosen_ids.shape[0]]
        rejected_logps = all_logps_sum[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_logps, rejected_logps, aux_loss, -all_logps_mean[: chosen_ids.shape[0]].mean()

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                return torch.cat(
                    [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks, prompt_id_lens * 2