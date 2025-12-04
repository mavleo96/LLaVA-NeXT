#!/usr/bin/env bash

###############################################################################
# Finetune LLaVA-NeXT with Qwen "alternating attention" locally
#
# This script is a simplified, local-machine friendly version of the
# OneVision finetune script. It assumes:
#   - single node
#   - 1–8 GPUs on the same machine
#   - your data is stored locally (edit the DATA_* paths below)
#
# Usage (examples):
#   chmod +x scripts/train/finetune_alternating_attn.sh
#   scripts/train/finetune_alternating_attn.sh
#   NUM_GPUS=4 RUN_NAME=llava-next-altattn scripts/train/finetune_alternating_attn.sh
#
# Important:
#   - The Qwen checkpoint path MUST contain "with_alternating_attn"
#     so that `llava/train/train.py` picks `LlavaQwenWithAlternatingAttnForCausalLM`.
#   - Edit DATA_YAML / IMAGE_FOLDER / VIDEO_FOLDER to match your local setup.
###############################################################################

set -euo pipefail

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

########################
## Model & Vision cfg ##
########################

# Qwen 0.5B alternating-attention LLaVA checkpoint from Hugging Face.
# This corresponds to the model used in `playground.py`:
#   model_path = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
#   model_name = "llava_qwen_with_alternating_attn"
#
# You can override this via:
#   LLM_VERSION=/path/to/your/local/ckpt scripts/train/finetune_alternating_attn.sh
# Default to the locally downloaded alternating-attn checkpoint.

IMAGE_FOLDER="/workspace/data/LLaVA-OneVision-Data"
RUN_NAME="llavanext-altattn-google_siglip-so400m-patch14-384-Qwen_Qwen2-0.5B-Instruct-local_finetune"
OUTPUT_DIR="/workspace/checkpoints/${RUN_NAME}"

#############################
## Launch training (single)##
#############################

NUM_GPUS=${NUM_GPUS:-2}

ACCELERATE_CPU_AFFINITY=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node "${NUM_GPUS}" \
  llava/train/train_mem.py \
  --attn_implementation "sdpa" \
  --model_name_or_path /workspace/checkpoints/llava-onevision-qwen2-0.5b-ov-with_alternating_attn \
  --version "qwen_1_5" \
  --data_path scripts/train/finetune_dataset.yaml \
  --image_folder "${IMAGE_FOLDER}" \
  --mm_tunable_parts "mm_language_model" \
  --mm_vision_tower_lr 0 \
  --vision_tower google/siglip-so400m-patch14-384 \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --group_by_modality_length True \
  --image_aspect_ratio anyres_max_9 \
  --image_grid_pinpoints "(1x1),...,(6x6)" \
  --mm_patch_merge_type spatial_unpad \
  --bf16 True \
  --run_name "${RUN_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --lora_enable True \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lora_bias "none" \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 1 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 False \
  --model_max_length 4096 \
  --gradient_checkpointing False \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to "none" \
  --torch_compile False \
  --dataloader_drop_last True \
  --frames_upbound 32

echo "Finetuning completed. Checkpoints are in: ${OUTPUT_DIR}"


