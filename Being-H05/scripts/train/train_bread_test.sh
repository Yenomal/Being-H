#!/bin/bash
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEINGH05_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${BEINGH05_ROOT}/.." && pwd)"

cd "${BEINGH05_ROOT}"

export PYTHONPATH=.
export NCCL_IB_DISABLE=0
export NO_ALBUMENTATIONS_UPDATE=1

# =============================================================================
# Required Paths - modify these first
# =============================================================================
# TODO: set these two paths to your local model directories before running.
PRETRAIN_MODEL="/path/to/InternVL3_5-2B"
EXPERT_MODEL="/path/to/Qwen3-0.6B"

# The repository already ships a base Being-H05 checkpoint under ../ckpt.
RESUME_ROOT="${REPO_ROOT}/ckpt"
RESUME_MODEL="Being-H05-2B"
RESUME_PATH="${RESUME_ROOT}/${RESUME_MODEL}"

# Bread smoke-test dataset config
DATASET_CONFIG_FILE="configs/posttrain/bread/bread.yaml"

# =============================================================================
# Optional: legacy bread data quaternion -> axis-angle preprocessing
# =============================================================================
# For the current legacy bread dataset, BreadDataConfig reads:
# - observation.state_axis_angle
# - action_axis_angle
#
# If these columns are already generated, you can leave this block disabled.
# If you need to regenerate them, set RUN_AXIS_ANGLE_PREP=true.
#
# When future datasets already store axis-angle directly in observation.state/action:
# 1. comment out STATE_SOURCE_COLUMN_OVERRIDE / ACTION_SOURCE_COLUMN_OVERRIDE in
#    configs/data_config.py::BreadDataConfig
# 2. keep RUN_AXIS_ANGLE_PREP=false here
RUN_AXIS_ANGLE_PREP=false
QUAT_ORDER="xyzw"
DATASET_ROOT="${REPO_ROOT}/datasets/lerobot/test_EE"

if [ "${RUN_AXIS_ANGLE_PREP}" = "true" ]; then
  python configs/convert_bread_quat_to_axis_angle.py \
    --dataset-root "${DATASET_ROOT}" \
    --quat-order "${QUAT_ORDER}" \
    --overwrite
fi

# =============================================================================
# Smoke Test Configuration
# =============================================================================
NUM_GPUS=1
MASTER_PORT=29116

MAX_STEPS=10
SAVE_STEPS=1000
SAVE_STEPS_START=100000
SAVE_MODEL_ONLY=True
SAVE_MERGED_META=True

LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-5
WARMUP_RATIO=0.01

NUM_WORKERS=0
PREFETCH_FACTOR=2

MAX_NUM_TOKENS=4096
EXPECTED_NUM_TOKENS=2048
PREFER_BUFFER_BEFORE=1024
MAX_BUFFER_SIZE=2
ATTN_MODE="causal"

FORCE_IMAGE_SIZE=224
MAX_VIEW_NUM=-1
USE_FIXED_VIEW=False
DOWN_SAMPLE_RATIO=0.5

ACTION_CHUNK_LENGTH=16

FREEZE_MLLM=False
FREEZE_VIT_MLP=False

# Use the simplest path for smoke test.
USE_MPG=False
MPG_LAMBDA=0.1
MPG_NUM_PROJECTIONS=32
MPG_REFINEMENT_ITERS=1
MPG_GATE_TEMPERATURE=1.0
MPG_USE_STOP_GRADIENT=True

USE_TRAINING_TIME_RTC=False
SIMULATED_DELAY=0
RTC_DELAY_EXP_WEIGHT=True
USE_INFERENCE_PREFIX_OVERWRITE=True

# =============================================================================
# Output
# =============================================================================
MODEL_NAME="smoke-bread_${RESUME_MODEL}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${REPO_ROOT}/output/bread_smoke/${MODEL_NAME}"
LOG_DIR="${REPO_ROOT}/output/tensorboard/bread_smoke"
LOG_FILE="${OUTPUT_DIR}/training.log"

# =============================================================================
# Sanity Checks
# =============================================================================
if [ ! -d "${PRETRAIN_MODEL}" ]; then
  echo "Error: PRETRAIN_MODEL does not exist: ${PRETRAIN_MODEL}"
  echo "Modify PRETRAIN_MODEL in ${0} before running."
  exit 1
fi

if [ ! -d "${EXPERT_MODEL}" ]; then
  echo "Error: EXPERT_MODEL does not exist: ${EXPERT_MODEL}"
  echo "Modify EXPERT_MODEL in ${0} before running."
  exit 1
fi

if [ ! -d "${RESUME_PATH}" ]; then
  echo "Error: RESUME_PATH does not exist: ${RESUME_PATH}"
  exit 1
fi

if [ ! -f "${DATASET_CONFIG_FILE}" ]; then
  echo "Error: DATASET_CONFIG_FILE not found: ${DATASET_CONFIG_FILE}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

cp "$0" "${OUTPUT_DIR}/"
mkdir -p "${OUTPUT_DIR}/code"
cp -r BeingH "${OUTPUT_DIR}/code/"

echo "=========================================="
echo "Bread smoke test training"
echo "BEINGH05_ROOT: ${BEINGH05_ROOT}"
echo "PRETRAIN_MODEL: ${PRETRAIN_MODEL}"
echo "EXPERT_MODEL: ${EXPERT_MODEL}"
echo "RESUME_PATH: ${RESUME_PATH}"
echo "DATASET_CONFIG_FILE: ${DATASET_CONFIG_FILE}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "=========================================="

# =============================================================================
# Launch Training
# =============================================================================
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=${NUM_GPUS} \
  --master_port=${MASTER_PORT} \
  BeingH/train/train.py \
  --mllm_path ${PRETRAIN_MODEL} \
  --expert_path ${EXPERT_MODEL} \
  --resume_from ${RESUME_PATH} \
  --resume_model_only True \
  --layer_module Qwen3MoTDecoderLayer \
  --use_expert True \
  --use_flow_matching True \
  --llm_qk_norm True \
  --freeze_mllm ${FREEZE_MLLM} \
  --freeze_vit_mlp ${FREEZE_VIT_MLP} \
  --action_chunk_length ${ACTION_CHUNK_LENGTH} \
  --max_num_tokens ${MAX_NUM_TOKENS} \
  --max_num_tokens_per_sample ${MAX_NUM_TOKENS} \
  --expected_num_tokens ${EXPECTED_NUM_TOKENS} \
  --prefer_buffer_before ${PREFER_BUFFER_BEFORE} \
  --max_buffer_size ${MAX_BUFFER_SIZE} \
  --attn_mode ${ATTN_MODE} \
  --max_view_num ${MAX_VIEW_NUM} \
  --use_fixed_view ${USE_FIXED_VIEW} \
  --force_image_size ${FORCE_IMAGE_SIZE} \
  --down_sample_ratio ${DOWN_SAMPLE_RATIO} \
  --dataset_config_file ${DATASET_CONFIG_FILE} \
  --save_merged_metadata ${SAVE_MERGED_META} \
  --conv_style "being_h0" \
  --vision_select_layer -1 \
  --prompt_template long \
  --output_dir ${OUTPUT_DIR} \
  --logging_dir ${LOG_DIR} \
  --num_workers ${NUM_WORKERS} \
  --prefetch_factor ${PREFETCH_FACTOR} \
  --max_steps ${MAX_STEPS} \
  --save_model_only ${SAVE_MODEL_ONLY} \
  --save_steps ${SAVE_STEPS} \
  --save_steps_start ${SAVE_STEPS_START} \
  --logging_steps 1 \
  --learning_rate ${LEARNING_RATE} \
  --weight_decay ${WEIGHT_DECAY} \
  --warmup_ratio ${WARMUP_RATIO} \
  --lr_scheduler cosine \
  --grad_checkpoint False \
  --gradient_accumulation_steps 1 \
  --use_mpg ${USE_MPG} \
  --mpg_lambda ${MPG_LAMBDA} \
  --mpg_num_projections ${MPG_NUM_PROJECTIONS} \
  --mpg_refinement_iters ${MPG_REFINEMENT_ITERS} \
  --mpg_gate_temperature ${MPG_GATE_TEMPERATURE} \
  --mpg_use_stop_gradient ${MPG_USE_STOP_GRADIENT} \
  --use_training_time_rtc ${USE_TRAINING_TIME_RTC} \
  --simulated_delay ${SIMULATED_DELAY} \
  --rtc_delay_exp_weight ${RTC_DELAY_EXP_WEIGHT} \
  --use_inference_prefix_overwrite ${USE_INFERENCE_PREFIX_OVERWRITE} \
  2>&1 | tee "${LOG_FILE}"

echo "=========================================="
echo "Bread smoke test complete"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "=========================================="
