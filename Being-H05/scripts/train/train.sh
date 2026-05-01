#!/bin/bash
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export PYTHONPATH=.
export NCCL_IB_DISABLE=0
export NO_ALBUMENTATIONS_UPDATE=1

# =============================================================================
# 路径配置
# =============================================================================
# 当前脚本默认从 Being-H 根目录启动：
#   bash Being-H05/scripts/train/train.sh
ROOT="$(pwd)"

# 项目根目录（Being-H05）
PROJECT_ROOT="${ROOT}/Being-H05"

# 当前脚本自身路径，用于保存复现实验脚本副本
SCRIPT_PATH="${ROOT}/Being-H05/scripts/train/train.sh"

# 基础多模态模型路径（InternVL 非 HF 格式）
PRETRAIN_MODEL="${ROOT}/ckpt/model/InternVL3_5-2B"

# Action Expert / LLM expert 路径
EXPERT_MODEL="${ROOT}/ckpt/model/Qwen3-0.6B"

# Being-H 预训练或上游 checkpoint 路径
RESUME_PATH="${ROOT}/ckpt/Being-H05-2B"

# 数据配置 YAML 路径
DATASET_CONFIG_FILE="${ROOT}/Being-H05/configs/posttrain/flower/flower.yaml"

# 输出根目录
OUTPUT_ROOT="${ROOT}/output/runs/flower"

# TensorBoard 日志根目录
LOG_ROOT="${ROOT}/output/tensorboard/flower"

cd "${PROJECT_ROOT}"

# =============================================================================
# GPU / 分布式配置
# =============================================================================
# 指定要使用的物理 GPU 编号列表
# 例如单卡：0
# 例如四卡：4,5,6,7
export CUDA_VISIBLE_DEVICES=0

# 使用多少张“可见 GPU”
# 这个值应与 CUDA_VISIBLE_DEVICES 里的卡数一致
NUM_GPUS=1

# torchrun 通信端口
MASTER_PORT=29111

# =============================================================================
# 训练步数与保存配置
# =============================================================================
# 总训练步数
MAX_STEPS=40000

# 每隔多少步保存一次 checkpoint
SAVE_STEPS=40000

# 从第多少步开始允许保存 checkpoint
SAVE_STEPS_START=0

# 是否只保存模型，不保存优化器和 scheduler
SAVE_MODEL_ONLY=True

# 是否保存 merged metadata
SAVE_MERGED_META=True

# =============================================================================
# 优化器配置
# =============================================================================
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-5
WARMUP_RATIO=0.05

# 累积梯度步数
# 想增大等效 batch、但又不想显存暴涨时可以调大
GRADIENT_ACCUMULATION_STEPS=1

# =============================================================================
# 数据加载配置
# =============================================================================
# DataLoader worker 数
NUM_WORKERS=32

# 每个 worker 预取 batch 数
PREFETCH_FACTOR=4

# =============================================================================
# Token packing / 等效 batch 配置
# =============================================================================
# 这套项目不是固定 batch_size，而是按 token 数动态组 batch
#
# MAX_NUM_TOKENS:
#   一个 packed batch 的硬上限；越大通常显存越高
#
# EXPECTED_NUM_TOKENS:
#   达到这个 token 数后，dataset 就倾向于把当前 batch 产出
#
# PREFER_BUFFER_BEFORE:
#   当前 batch token 数低于这个值时，优先从 buffer 取样本而不是新采样
#
# MAX_BUFFER_SIZE:
#   buffer 最多缓存多少个样本
MAX_NUM_TOKENS=8704
EXPECTED_NUM_TOKENS=8192
PREFER_BUFFER_BEFORE=4096
MAX_BUFFER_SIZE=4

# 注意力模式
ATTN_MODE="causal"

# =============================================================================
# 图像 / 视角配置
# =============================================================================
# 输入图像尺寸
FORCE_IMAGE_SIZE=448

# 最多使用多少个视角
# -1 表示使用 DataConfig 中的全部视角
MAX_VIEW_NUM=-1

# 是否固定只使用一个视角
USE_FIXED_VIEW=True

# ViT token 下采样比例
DOWN_SAMPLE_RATIO=0.5

# =============================================================================
# 动作生成配置
# =============================================================================
# 每次预测多少步动作
ACTION_CHUNK_LENGTH=16

# =============================================================================
# 冻结配置
# =============================================================================
FREEZE_MLLM=False
FREEZE_VIT_MLP=False

# =============================================================================
# MPG 配置
# =============================================================================
# 是否启用 MPG
USE_MPG=False
MPG_LAMBDA=0.1
MPG_NUM_PROJECTIONS=32
MPG_REFINEMENT_ITERS=1
MPG_GATE_TEMPERATURE=1.0
MPG_USE_STOP_GRADIENT=True

# =============================================================================
# RTC 配置
# =============================================================================
USE_TRAINING_TIME_RTC=True
SIMULATED_DELAY=8
RTC_DELAY_EXP_WEIGHT=True
USE_INFERENCE_PREFIX_OVERWRITE=True

# =============================================================================
# 输出目录
# =============================================================================
MODEL_NAME="flower_delta_head"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}"
LOG_DIR="${LOG_ROOT}"
LOG_FILE="${OUTPUT_DIR}/training.log"

# =============================================================================
# 检查路径
# =============================================================================
if [ ! -d "${PRETRAIN_MODEL}" ]; then
  echo "Error: PRETRAIN_MODEL does not exist: ${PRETRAIN_MODEL}"
  exit 1
fi

if [ ! -d "${EXPERT_MODEL}" ]; then
  echo "Error: EXPERT_MODEL does not exist: ${EXPERT_MODEL}"
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

cp "${SCRIPT_PATH}" "${OUTPUT_DIR}/"
mkdir -p "${OUTPUT_DIR}/code"
cp -r BeingH "${OUTPUT_DIR}/code/"

echo "=========================================="
echo "Bread training"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "SCRIPT_PATH: ${SCRIPT_PATH}"
echo "PRETRAIN_MODEL: ${PRETRAIN_MODEL}"
echo "EXPERT_MODEL: ${EXPERT_MODEL}"
echo "RESUME_PATH: ${RESUME_PATH}"
echo "DATASET_CONFIG_FILE: ${DATASET_CONFIG_FILE}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "=========================================="

# =============================================================================
# 启动训练
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
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
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
echo "Training complete"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "=========================================="
