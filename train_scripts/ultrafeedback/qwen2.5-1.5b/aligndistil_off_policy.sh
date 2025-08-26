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

# boost 调参
task=ultrafeedback
model=qwen2.5-1.5b
method=aligndistil
kd_divergence=rkl
batch_size=128
lr=1e-6
kd_coef=1.0
beta=0.1
beta2=10

task_config=off_policy_beta${beta}_beta2_${beta2}_lr${lr}_bsz${batch_size}
output_dir=$work_dir/checkpoint/$task/$model/$method/$task_config
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

deepspeed --num_gpus 8 $code_dir/openrlhf/cli/train_aligndistil_off_policy.py \
   --pretrain $model_path \
   --teacher_model $teacher_model_path \
   --ref_model $ref_model_path \
   --kd_coef $kd_coef \
   --kd_divergence $kd_divergence \
   --beta $beta \
   --beta2 $beta2 \
   --reward_boost_type "aligndistil" \
   --reward_len_norm \
   --dataset $data_path \
   --train_split train_prefs \
   --chosen_key chosen \
   --rejected_key rejected \
   --apply_chat_template \
   --train_batch_size $batch_size \
   --micro_train_batch_size 1 \
   --learning_rate $lr \
   --warmup_ratio 0.1 \
   --save_path $output_dir \
   --save_steps -1 \
   --logging_steps 10 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --flash_attn \
   --bf16 \
   --gradient_checkpointing 2>&1 | tee -a $output_dir/train.log

