# Pretrain script
export MODEL_NAME="/path/to/CogVideoX-5b"
export DATASET_META_NAME="/path/to/your_own_pretrain_dataset.csv"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
echo ${PORT}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
GPU_NUM_PER_NODE=${GPU_NUM_PER_NODE:-8}
GPUS=`expr $GPU_NUM_PER_NODE \* ${NNODES}`
echo "train on ${GPUS} GPUS"

accelerate launch \
  --num_processes ${GPUS} \
  --num_machines ${NNODES} \
  --main_process_port ${PORT} \
  --machine_rank ${NODE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --use_deepspeed \
  --deepspeed_config_file config/zero_stage2_config.json \
  --deepspeed_multinode_launcher standard scripts/cogvideox_fun/train_mask2dit.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1024 \
  --video_sample_size 480 720 \
  --token_sample_size=512 \
  --video_sample_stride=3 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --num_scenes=3 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=1000 \
  --learning_rate=1e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir/5b/pretrain" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --use_deepspeed \
  --attn_msk_version 2 \
  --trainable_modules "."
