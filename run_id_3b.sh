export PJRT_DEVICE=TPU
export XLA_USE_SPMD=1
export XLA_USE_BF16=1
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
export LIBTPU_INIT_ARGS="--xla_enable_async_collective_permute=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_jf_spmd_threshold_for_windowed_einsum_mib=0"
export HF_DATASETS_CACHE=~/model/HF/
export PROFILE_EPOCH=0
export PROFILE_STEP=3
export PROFILE_DURATION_MS=20000
export PROFILE_LOGDIR=~/log
python examples/pytorch/language-modeling/run_clm.py \
  --model_name_or_path /home/hyun/Qwen3-0.6B \
  --tokenizer_name Qwen/Qwen2-0.5B \
  --dataset_name dkqjrm/korean-english-qwen-templated-small \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1 \
  --save_steps 10 \
  --logging_steps 1 \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --config_name /home/hyun/Qwen3-0.6B \
  --output_dir ~/result \
  --overwrite_output_dir \
  --remove_unused_columns no \
  --optim adafactor \
  --torch_dtype bfloat16 \
  --dataloader_drop_last yes \
  --block_size 4096 \
  --spmd_2d_sharding 16 \
  --spmd_grad_chkpt \
  --report_to wandb \
  --wandb_key $WANDB_API_KEY \
  --peft_lora \
  --lora_rank 32 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.1