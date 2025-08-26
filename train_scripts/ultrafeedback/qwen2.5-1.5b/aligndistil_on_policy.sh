export HF_HOME="/path/to/huggingface/cache"  # only set if you need
export LD_LIBRARY_PATH="/usr/lib64/":$LD_LIBRARY_PATH
set -x

work_dir=/path/to/AlignDistil
code_dir=$work_dir/OpenRLHF
export PYTHONPATH=$code_dir:$PYTHONPATH

data_path=$work_dir/data/ultrafeedback_binarized
model_path=/path/to/qwen2.5-1.5b-instruct
teacher_model_path=/path/to/dpo_model
ref_model_path=/path/to/reverse_dpo_model

task=ultrafeedback
model=qwen2.5-1.5b
method=aligndistil
kd_divergence=rkl
kd_temperature=1.0
batch_size=128
lr=1e-6
kd_coef=1.0
beta=0.1
beta2=15
constrant=teacher
n_samples_per_prompt=1

task_config=on_policy_beta${beta}_beta2_${beta2}_lr${lr}_bsz${batch_size}
output_dir=$work_dir/checkpoint/$task/$model/$method/$task_config
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

torchrun $code_dir/openrlhf/cli/train_aligndistil_on_policy.py \
   --student_num_nodes 1 \
   --student_num_gpus_per_node 4 \
   --teacher_num_nodes 1 \
   --teacher_num_gpus_per_node 2 \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --colocate_teacher_ref \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --vllm_sync_steps 1 \
   --pretrain $model_path \
   --teacher_model $teacher_model_path \
   --ref_model $ref_model_path \
   --kd_coef $kd_coef \
   --kd_divergence $kd_divergence \
   --kd_temperature $kd_temperature \
   --reward_boost_type "aligndistil" \
   --reward_len_norm \
   --beta $beta \
   --beta2 $beta2 \
   --dataset $data_path \
   --train_split train_prefs \
   --input_key messages \
   --apply_chat_template \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --train_batch_size $batch_size \
   --micro_train_batch_size 1 \
   --micro_prepare_batch_size 1 \
   --learning_rate $lr \
   --warmup_ratio 0.1 \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size $batch_size \
   --n_samples_per_prompt $n_samples_per_prompt \
   --temperature 1.0 \
   --top_p 1.0 \
   --save_path $output_dir \
   --save_steps -1 \
   --logging_steps 10 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --flash_attn \
   --bf16 \
   --gradient_checkpointing 2>&1 | tee -a $output_dir/train.log
