#!/bin/bash
set -euo pipefail

MASTER_PORT=$((RANDOM % 101 + 20001))
REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# --- Data & checkpoints (edit these paths for your machine) ---
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
VIDEO_FOLDER="${VIDEO_FOLDER:-${DATA_ROOT}/trajectory_data/R2R,${DATA_ROOT}/trajectory_data/RxR,${DATA_ROOT}/trajectory_data/EnvDrop}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/prospect}"
PREV_STAGE_CHECKPOINT="${PREV_STAGE_CHECKPOINT:-lmms-lab/LLaVA-Video-7B-Qwen2}"

# --- Model: CUT3R spatial encoder + latent 2D/3D world model ---
USE_WORLD_QUERY="True"
NUM_WORLD_QUERY_TOKENS=9
WORLD_QUERY_DECODER_DEPTH=2
WM_LOSS_WEIGHT=0.01
WM_2D_LOSS_RATIO=0.5
WM_USE_2D_LOSS="True"
WM_USE_3D_LOSS="True"
WM_2D_LOSS_TYPE="cosine"
MM_TUNABLE_PARTS="mm_vision_tower,mm_mlp_adapter,mm_language_model,fusion_block,world_model"
RUN_NAME="${RUN_NAME:-prospect}"

DEEPSPEED_CONFIG="${REPO_ROOT}/scripts/zero2.json"
if [[ ! -f "$DEEPSPEED_CONFIG" ]]; then
  echo "Error: DeepSpeed config not found: ${DEEPSPEED_CONFIG}"
  exit 1
fi

SIGLIP_MODEL="google/siglip-so400m-patch14-384"
PROMPT_VERSION="qwen_1_5"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NPROC=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
else
  NPROC=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n1)
  if [[ -z "$NPROC" || "$NPROC" -eq 0 ]]; then
    echo "Error: no GPU detected"
    exit 1
  fi
fi

echo "Training with ${NPROC} GPU(s)"
echo "Output: ${OUTPUT_DIR}"
echo "Base model: ${PREV_STAGE_CHECKPOINT}"
echo "Trajectory data: ${VIDEO_FOLDER}"

torchrun --nproc_per_node=${NPROC} \
    --master_port ${MASTER_PORT} \
    -m streamvln.streamvln_train \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path "${PREV_STAGE_CHECKPOINT}" \
    --version ${PROMPT_VERSION} \
    --video_folder "${VIDEO_FOLDER}" \
    --group_by_task False \
    --num_history 8 \
    --num_future_steps 4 \
    --num_frames 32 \
    --data_augmentation True \
    --mm_tunable_parts="${MM_TUNABLE_PARTS}" \
    --vision_tower ${SIGLIP_MODEL} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --use_world_query ${USE_WORLD_QUERY} \
    --num_world_query_tokens ${NUM_WORLD_QUERY_TOKENS} \
    --world_query_decoder_depth ${WORLD_QUERY_DECODER_DEPTH} \
    --wm_loss_weight ${WM_LOSS_WEIGHT} \
    --wm_2d_loss_ratio ${WM_2D_LOSS_RATIO} \
    --wm_use_2d_loss ${WM_USE_2D_LOSS} \
    --wm_use_3d_loss ${WM_USE_3D_LOSS} \
    --wm_2d_loss_type ${WM_2D_LOSS_TYPE} \
    --bf16 True \
    --run_name ${RUN_NAME} \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --mm_vision_tower_lr 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.075 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr": 1.85e-05}' \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --report_to none
