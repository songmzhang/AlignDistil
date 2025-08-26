export HF_HOME="/path/to/huggingface/cache"  # only set if you need
export LD_LIBRARY_PATH="/usr/lib64/":$LD_LIBRARY_PATH
set -x

work_dir=/path/to/AlignDistil
code_dir=$work_dir/OpenRLHF
export PYTHONPATH=$code_dir:$PYTHONPATH

data_path=$work_dir/data/ultrafeedback_binarized/
pretrain_model_path=/path/to/qwen2.5-1.5b-instruct

task=ultrafeedback
model=qwen2.5-1.5b
method=dpo
batch_size=128
lr=1e-6
beta=0.01
sft_coef=0.0
epochs=1

task_config=dpo_beta${beta}_sft${sft_coef}_lr${lr}_bsz${batch_size}
output_dir=$work_dir/checkpoint/$task/$model/$method/$task_config
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi


deepspeed --num_gpus 8 $code_dir/openrlhf/cli/train_dpo.py \
   --dataset $data_path \
   --train_split train_prefs \
   --chosen_key chosen \
   --rejected_key rejected \
   --apply_chat_template \
   --max_len 2048 \
   --train_batch_size $batch_size \
   --micro_train_batch_size 1 \
   --learning_rate $lr \
   --warmup_ratio 0.1 \
   --beta $beta \
   --nll_loss_coef $sft_coef \
   --pretrain $pretrain_model_path \
   --save_path $output_dir \
   --save_steps -1 \
   --logging_steps 10 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs $epochs \
   --bf16 \
   --flash_attn \
   --gradient_checkpointing 2>&1 | tee -a $output_dir/train.log
