import argparse
import math
import os
import multiprocessing
from datetime import datetime

import ray
import torch
from ray.util.placement_group import placement_group

from openrlhf.trainer.ray import (
    KDRayActorGroup,
    StudentModelRayActor,
    TeacherModelRayActor,
    ReferenceModelRayActor,
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
    # _validate_args(args)
    
    # configure strategy
    strategy = get_strategy(args)
    # strategy.setup_distributed()
    
    if args.colocate_student_teacher_ref:
        assert (
            args.student_num_nodes == args.ref_num_nodes and args.student_num_gpus_per_node == args.ref_num_gpus_per_node
        ), f"num_nodes and num_gpus_per_node must be the same when colocate ref and student model."
        assert (
            args.student_num_nodes == args.teacher_num_nodes and args.student_num_gpus_per_node == args.teacher_num_gpus_per_node
        ), f"num_nodes and num_gpus_per_node must be the same when colocate teacher and student model."

        bundles = [
            {"GPU": args.student_num_gpus_per_node, "CPU": args.student_num_gpus_per_node}
            for _ in range(args.student_num_nodes)
        ]
        pg = placement_group(bundles, strategy="SPREAD")
        ray.get(pg.ready())
    
        student_model = KDRayActorGroup(
            args.student_num_nodes,
            args.student_num_gpus_per_node,
            StudentModelRayActor,
            pg=pg,
            num_gpus_per_actor=0.4
        )
        teacher_model = KDRayActorGroup(
            args.teacher_num_nodes,
            args.teacher_num_gpus_per_node,
            TeacherModelRayActor,
            pg=pg,
            num_gpus_per_actor=0.31
        )
        
        ref_model = KDRayActorGroup(
            args.ref_num_nodes,
            args.ref_num_gpus_per_node,
            TeacherModelRayActor,     # use TeacherModelRayActor here for full log_prob distribution
            pg=pg,
            num_gpus_per_actor=0.29
        )
    else:
        student_model = KDRayActorGroup(
            args.student_num_nodes,
            args.student_num_gpus_per_node,
            StudentModelRayActor,
            pg=None,
            num_gpus_per_actor=1
        )
        pg = None
        # if colocated, create placement group for actor and ref model explicitly.
        if args.colocate_teacher_ref:
            assert (
                args.teacher_num_nodes == args.ref_num_nodes and args.teacher_num_gpus_per_node == args.ref_num_gpus_per_node
            ), f"num_nodes and num_gpus_per_node must be the same when colocate teacher and ref model."

            bundles = [
                {"GPU": args.teacher_num_gpus_per_node, "CPU": args.teacher_num_gpus_per_node}
                for _ in range(args.teacher_num_nodes)
            ]
            pg = placement_group(bundles, strategy="STRICT_SPREAD")
            ray.get(pg.ready())

        # NOTE(wuxibin): Why don't we allocate 0.5 gpu for each actor when colocate models?
        # Say we have 1 node with 4 GPUs, and num_gpus_per_node for each model is 4.
        # If we allocate 0.5 gpu for both actor and ref model, then gpu allocation is
        #   |actor|actor|actor|actor|  ref | ref  | ref  | ref |
        #   |GPU0 |GPU0 |GPU1 |GPU1 | GPU2 | GPU2 | GPU3 | GPU3 |
        #
        # So 0.75/0.25 gpu is a tricky to let Ray spread all models evenly on all gpus.
        #   |actor| ref  |actor| ref  |actor| ref  |actor|ref  |
        #   |GPU0 | GPU0 |GPU1 | GPU1 |GPU2 | GPU2 |GPU3 | GPU3 |
        
        teacher_model = KDRayActorGroup(
            args.teacher_num_nodes,
            args.teacher_num_gpus_per_node,
            TeacherModelRayActor,
            pg=pg,
            num_gpus_per_actor=0.51 if pg else 1
        )
        
        ref_model = KDRayActorGroup(
            args.ref_num_nodes,
            args.ref_num_gpus_per_node,
            TeacherModelRayActor,     # use TeacherModelRayActor here for full log_prob distribution
            pg=pg,
            num_gpus_per_actor=0.49 if pg else 1,
        )
    
    # init student/teacher model
    refs = []
    refs.extend(student_model.async_init_model_from_pretrained(strategy, args.pretrain))
    refs.extend(teacher_model.async_init_model_from_pretrained(strategy, args.teacher_model))
    if args.ref_model is None:
        refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.pretrain))
    else:
        refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.ref_model))
    
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
        teacher_model_group=teacher_model, 
        ref_model_group=ref_model,
        vllm_engines=vllm_engines
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
    parser.add_argument("--ref_num_nodes", type=int, default=1, help="number of nodes for reference model")
    parser.add_argument("--ref_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reference model")
    parser.add_argument(
        "--colocate_student_teacher_ref",
        action="store_true",
        default=False,
        help="whether to colocate student, reference and teacher model, if true, they will share same gpus.",
    )
    parser.add_argument(
        "--colocate_teacher_ref",
        action="store_true",
        default=False,
        help="whether to colocate reference and teacher model, if true, they will share same gpus.",
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
    parser.add_argument("--ref_model", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--kd_coef", type=float, default=0.4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--pretrain_mode", action="store_true", default=False, help="Use pretrain loss")
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--l2", type=float, default=0, help="weight decay loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    parser.add_argument("--teacher_offload", action="store_true", default=False)
    parser.add_argument("--static_model_stage", type=int, default=0)
    parser.add_argument("--kd_divergence", type=str, default="forward_kl")
    parser.add_argument("--kd_temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--beta2", type=float, default=1.0)
    parser.add_argument("--reward_type", type=str, default="dpo")
    parser.add_argument("--topk_reward", type=int, default=-1)
    parser.add_argument("--micro_prepare_batch_size", type=int, default=2)
    parser.add_argument("--trainer", type=str, default="constrant_policy_kd")
    parser.add_argument("--clamp_max_ref_rkl", action="store_true", default=False, help="clip max kl with reference")
    parser.add_argument("--max_ref_rkl", type=float, default=None)
    parser.add_argument("--control_max_dpo_rkl", action="store_true", default=False, help="max dpo kl not exceed some value")
    parser.add_argument("--max_dpo_rkl", type=float, default=2.0)
    parser.add_argument("--constrant", type=str, default="teacher")
    parser.add_argument("--reward_len_norm", action="store_true", default=False, help="Use length normalized reward")
    parser.add_argument("--reward_len_norm_coef", type=float, default=1.0, help="Length normalized coefficient")
    parser.add_argument("--length_penalty", action="store_true", default=False, help="Use length penalty to reward")
    parser.add_argument("--length_penalty_coef", type=float, default=0.002, help="Length penalty coefficient")
    parser.add_argument("--reward_temperature", type=float, default=1.0, help="Temperature for DPO reward")
    parser.add_argument("--min_p", type=float, default=1e-4, help="Min probability for DPO reward")
    parser.add_argument("--vllm_sync_steps", type=int, default=1, help="Sync vllm weights every n steps")
    parser.add_argument("--remain_topk_reward", type=int, default=10, help="Remain top-k reward and last-k reward")
    parser.add_argument("--combine_coef", type=float, default=0.0, help="Coefficient for combining ref and teacher")
    parser.add_argument("--reward_clip", type=float, default=None, help="Reward clip")
    parser.add_argument("--reward_shape", type=str, default="distribution_reward_kl", help="Reward type")
    parser.add_argument("--self_neg_reward", action="store_true", default=False, help="Use self negative contrast reward")
    parser.add_argument("--neg_ratio", type=float, default=1.0, help="Ratio for self negative contrast reward")
    parser.add_argument("--boost_dpo_reward", action="store_true", default=False, help="Use boost dpo reward")
    parser.add_argument("--pos_ratio", type=float, default=1.0, help="Ratio for boost dpo reward")
    parser.add_argument("--del_reward_from", type=str, default=None, help="Ratio for boost dpo reward")
    parser.add_argument("--reward_del_thres", type=float, default=0.5, help="Ratio for boost dpo reward")
    parser.add_argument("--reward_boost_type", type=str, default=None, help="Ratio for boost dpo reward")
    parser.add_argument("--add_value_head", action="store_true", default=False)
    
    # Generation
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--generate_max_len", type=int, default=None, help="Max tokens to generate")
    parser.add_argument("--prompt_max_len", type=int, default=None, help="Max tokens for each prompt")
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
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
    print(args)
    multiprocessing.set_start_method('spawn')
    train(args)
