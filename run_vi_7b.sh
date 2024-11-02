export PJRT_DEVICE=TPU
export XLA_USE_SPMD=1
export XLA_USE_BF16=1
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
export LIBTPU_INIT_ARGS="--xla_enable_async_collective_permute=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_jf_spmd_threshold_for_windowed_einsum_mib=0"
export HF_DATASETS_CACHE=/model/HF/
export PROFILE_EPOCH=0
export PROFILE_STEP=3
export PROFILE_DURATION_MS=20000
export PROFILE_LOGDIR=/tmp/home/
sudo python examples/pytorch/language-modeling/run_clm.py --model_name_or_path /mnt/nfs_share/qwen25_7b_it --tokenizer_name /mnt/nfs_share/qwen25_7b_it --dataset_name jaeyong2/vi_sample_sft_data --per_device_train_batch_size 1 --per_device_eval_batch_size 8 --gradient_accumulation_steps 1 --num_train_epochs 1 --save_steps 400000 --do_train --config_name /mnt/nfs_share/qwen25_7b_it/config.json --output_dir /mnt/nfs_share/model25_qwen7b_vi --overwrite_output_dir --remove_unused_columns no --optim adafactor --learning_rate 1e-6 --torch_dtype bfloat16 --dataloader_drop_last yes --block_size 3072 --spmd_2d_sharding 16 --spmd_grad_chkpt
