import argparse
import math
import os
from datetime import datetime

import ray
import torch
from ray.util.placement_group import placement_group

from openrlhf.trainer.ray import (
    KDRayActorGroup,
    StudentModelRayActor,
    TeacherModelRayActor,
    create_vllm_engines,
)

from transformers.trainer import get_scheduler

from openrlhf.datasets import SFTDataset
from openrlhf.models import Actor
from openrlhf.trainer import OnPolicyKDTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer


def _validate_args(args):
    student_world_size = args.student_num_nodes * args.student_num_gpus_per_node

    assert (
        student_world_size & (student_world_size - 1)
    ) == 0, f"student_world_size must be power of 2, got {student_world_size}"

    assert args.zero_stage != 3 or args.vllm_num_engines > 0, f"ZeRO-3 is only supported when vLLM enabled"


def train(args):
    _validate_args(args)
    
    # configure strategy
    strategy = get_strategy(args)
    # strategy.setup_distributed()
    
    if args.colocate_student_teacher:
        assert (
            args.student_num_nodes == args.teacher_num_nodes and args.student_num_gpus_per_node == args.teacher_num_gpus_per_node
        ), f"num_nodes and num_gpus_per_node must be the same when colocate teacher and student model."

        bundles = [
            {"GPU": args.student_num_gpus_per_node, "CPU": args.student_num_gpus_per_node}
            for _ in range(args.student_num_nodes)
        ]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(pg.ready())
    
        student_model = KDRayActorGroup(
            args.student_num_nodes,
            args.student_num_gpus_per_node,
            StudentModelRayActor,
            pg=pg,
            num_gpus_per_actor=0.51
        )
        teacher_model = KDRayActorGroup(
            args.teacher_num_nodes,
            args.teacher_num_gpus_per_node,
            TeacherModelRayActor,
            pg=pg,
            num_gpus_per_actor=0.49
        )
    else:
        student_model = KDRayActorGroup(
            args.student_num_nodes,
            args.student_num_gpus_per_node,
            StudentModelRayActor,
            pg=None,
            num_gpus_per_actor=1
        )
        teacher_model = KDRayActorGroup(
            args.teacher_num_nodes,
            args.teacher_num_gpus_per_node,
            TeacherModelRayActor,
            pg=None,
            num_gpus_per_actor=1
        )
    
    # init student/teacher model
    refs = []
    refs.extend(student_model.async_init_model_from_pretrained(strategy, args.pretrain))
    refs.extend(teacher_model.async_init_model_from_pretrained(strategy, args.teacher_model))
    
    ray.get(refs)

    # init vLLM engine for text generation
    vllm_engines = None
    if args.vllm_num_engines is not None and args.vllm_num_engines > 0:
        max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        vllm_engines = create_vllm_engines(
            args.vllm_num_engines,
            args.vllm_tensor_parallel_size,
            args.pretrain,
            args.seed,
            args.enable_prefix_caching,
            args.enforce_eager,
            max_len,
        )
    
    # train student model
    refs = student_model.async_fit_kd(
        teacher_model, vllm_engines=vllm_engines
    )
    ray.get(refs)
    
    # save model
    ray.get(student_model.async_save_model())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Ray and vLLM
    parser.add_argument("--student_num_nodes", type=int, default=1, help="number of nodes for student model")
    parser.add_argument("--student_num_gpus_per_node", type=int, default=8, help="number of gpus per node for student model")
    parser.add_argument("--teacher_num_nodes", type=int, default=1, help="number of nodes for teacher model")
    parser.add_argument("--teacher_num_gpus_per_node", type=int, default=8, help="number of gpus per node for teacher model")
    parser.add_argument(
        "--colocate_student_teacher",
        action="store_true",
        default=False,
        help="whether to colocate student and teacher model, if true, they will share same gpus.",
    )
    # optional vLLM for text generation
    parser.add_argument(
        "--vllm_num_engines", type=int, default=None, help="number of vLLM Engines, set to 0 to disable vLLM"
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size of vLLM Engine for multi-GPU inference",
    )
    parser.add_argument("--vllm_sync_backend", type=str, default="gloo", help="DeepSpeed -> vLLM weight sync backend")
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--enforce_eager", action="store_true", default=False, help="Disable CUDA graph in vLLM")
    
    # Checkpoints
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_kd")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--micro_train_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--static_model_stage", type=int, default=0)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # KD
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--teacher_model", type=str, default=None)
    parser.add_argument("--trainer", type=str, default="constrant_policy_kd")
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--kd_coef", type=float, default=0.4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--pretrain_mode", action="store_true", default=False, help="Use pretrain loss")
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--l2", type=float, default=0, help="weight decay loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    parser.add_argument("--teacher_offload", action="store_true", default=False)
    parser.add_argument("--kd_divergence", type=str, default="forward_kl")
    parser.add_argument("--kd_temperature", type=float, default=1.0)
    parser.add_argument("--add_value_head", action="store_true", default=False)
    
    # Generation
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--generate_max_len", type=int, default=None, help="Max tokens to generate")
    parser.add_argument("--prompt_max_len", type=int, default=None, help="Max tokens for each prompt")
    parser.add_argument("--micro_rollout_batch_size", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=128)
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["gae", "reinforce"],
        default="gae",
        help="Choose advantage estimation method: gae, reinforce",
    )

    # Custom dataset
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--eval_split", type=str, default="test", help="test split of the dataset")

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default="User: {}\nAssistant: ")
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )

    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_sft")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="sft_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    args = parser.parse_args()
    train(args)
