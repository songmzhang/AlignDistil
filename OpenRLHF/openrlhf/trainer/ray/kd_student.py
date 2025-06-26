import itertools
import math
import os
import socket
from typing import Callable, Dict, List

import deepspeed
import ray
import torch
from transformers.trainer import get_scheduler

from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.models import Actor
from openrlhf.trainer import AlignDistilTrainer
from openrlhf.trainer.ppo_utils import Experience, RemoteExperienceMaker
from openrlhf.utils import blending_datasets, get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.distributed_util import init_process_group

from .launcher import BasePPORole


@ray.remote(num_gpus=1)
class StudentModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        args = strategy.args
        self._setup_distributed(strategy)

        actor = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            packing_samples=strategy.args.packing_samples,
        )
        if args.add_value_head:
            actor.model.add_module("value_head", torch.nn.Linear(actor.model.config.hidden_size, actor.model.config.vocab_size, bias=False))
            torch.nn.init.constant_(actor.model.value_head.weight, 0.0)
            
        strategy.print(actor)

        # configure tokenizer
        self.tokenizer = get_tokenizer(
            pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
        )

        # configure optimizer
        actor_optim = strategy.create_optimizer(
            actor, lr=args.learning_rate, betas=strategy.args.adam_betas, weight_decay=args.l2
        )

        # prepare_datasets
        self.prepare_datasets()

        # configure scheduler
        self.num_update_steps_per_epoch = len(self.prompts_dataset) // args.train_batch_size
        max_steps = math.ceil(args.max_epochs * self.num_update_steps_per_epoch)
        self._max_steps = max_steps

        actor_scheduler = get_scheduler(
            args.lr_scheduler,
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * args.warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
        )

        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.model, self.optim, self.scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler)
        )

        # load checkpoint
        self.consumed_samples = 0
        if args.load_checkpoint and os.path.exists(args.ckpt_path):
            _, states = strategy.load_ckpt(self.actor.model, args.ckpt_path)
            self.consumed_samples = states["consumed_samples"]
            strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {self.consumed_samples}")

    def prepare_datasets(self):
        strategy = self.strategy
        args = self.strategy.args

        # prepare for data and dataset
        # prepare datasets
        prompts_data = blending_datasets(
            args.dataset,
            args.dataset_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            return_eval=False,
            train_split=args.train_split,
        )
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
        self.prompts_dataset = PromptDataset(
            prompts_data, self.tokenizer, strategy, input_template=args.input_template
        )
        self.train_dataloader = strategy.setup_dataloader(
            self.prompts_dataset, args.rollout_batch_size // strategy.world_size, True, True
        )
        # train_data, eval_data = blending_datasets(
        #     args.dataset,
        #     args.dataset_probs,
        #     strategy,
        #     args.seed,
        #     max_count=args.max_samples,
        #     train_split=args.train_split,
        #     eval_split=args.eval_split,
        # )
        # train_data = train_data.select(range(min(args.max_samples, len(train_data))))
        # eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
        # self.train_dataset = SFTDataset(
        #     train_data,
        #     self.tokenizer,
        #     args.max_len,
        #     strategy,
        #     pretrain_mode=args.pretrain_mode,
        #     input_template=args.input_template,
        # )
        # self.eval_dataset = SFTDataset(
        #     eval_data,
        #     self.tokenizer,
        #     args.max_len,
        #     strategy,
        #     pretrain_mode=args.pretrain_mode,
        #     input_template=args.input_template,
        # )

        # self.train_dataloader = strategy.setup_dataloader(
        #     self.train_dataset, args.micro_train_batch_size, True, True, self.train_dataset.collate_fn
        # )
        # self.eval_dataloader = strategy.setup_dataloader(
        #     self.eval_dataset, args.micro_train_batch_size, True, False, self.eval_dataset.collate_fn
        # )

    def max_steps(self):
        """Return the maximum number of steps."""
        return self._max_steps

    def fit(
        self,
        teacher_model: ray.actor.ActorHandle,
        ref_model: ray.actor.ActorHandle = None,
        vllm_engines: List[ray.actor.ActorHandle] = None,
    ):
        """Train actor model with prompt datasets."""
        strategy = self.strategy
        args = self.strategy.args
        Trainer = AlignDistilTrainer

        # configure Trainer
        trainer = Trainer(
            model=self.model,
            teacher_model=teacher_model,
            ref_model=ref_model,
            strategy=strategy,
            optim=self.optim,
            train_dataloader=self.train_dataloader,
            eval_dataloader=None,
            scheduler=self.scheduler,
            max_norm=args.max_norm,
            pretrain_mode=args.pretrain_mode,
            batch_size=args.train_batch_size,
            max_epochs=args.max_epochs,
            tokenizer=self.tokenizer,
            # for LLM generation
            do_sample=True,
            max_new_tokens=args.generate_max_len,
            max_length=args.max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            vllm_engines=vllm_engines,
        )

        # broadcast checkpoint
        if args.load_checkpoint and os.path.exists(args.ckpt_path) and not vllm_engines is None:
            torch.distributed.barrier()
            trainer._broadcast_to_vllm()

        trainer.fit(
            args,
            self.consumed_samples,
            self.num_update_steps_per_epoch,
        )

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.model,
            self.tokenizer,
            args.save_path,
        )
