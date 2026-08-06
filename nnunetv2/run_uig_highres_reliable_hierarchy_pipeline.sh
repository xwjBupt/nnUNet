#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NNUNET_ENV_BIN="${NNUNET_ENV_BIN:-/home/wjx/miniconda3/envs/nnunet_seg/bin}"

export PATH="${NNUNET_ENV_BIN}:${PATH}"
export nnUNet_raw="${nnUNet_raw:-/home/wjx/CodeData/data/nnUNetData/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/home/wjx/CodeData/data/nnUNetData/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/home/wjx/CodeData/code/nnUNet/nnUNet_results}"

# This experiment intentionally uses the pre-51eed63 eager execution path.
# nnU-Net enables compile when this variable is absent, so it must be false.
export nnUNet_compile=false

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
CONFIG_NAME="segmamba_uig_dec2_logit_boundary_hierarchy_highres2_reliable_128x96x96"
TRAINER_NAME="nnUNetTrainerSegMambaUIGStableHierarchyHighResReliable"

"${NNUNET_ENV_BIN}/python" \
  "${REPO_DIR}/nnunetv2Extention/SegMamba_V2_UIG/prepare_highres_reliable_hierarchy_plans.py" \
  --dataset-name Dataset515_ICH2023 \
  --batch-size "${TRAIN_BATCH_SIZE}"

if [ "${PREPARE_ONLY:-0}" = "1" ]; then
  echo "Plans preparation completed; PREPARE_ONLY=1, pipeline not started."
  exit 0
fi

DATASET_ID=515 \
PLANS_NAME=nnUNetPlans_segmamba_ui \
CONFIG_NAME="${CONFIG_NAME}" \
TRAINER_NAME="${TRAINER_NAME}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6,7}" \
NUM_THREADS="${NUM_THREADS:-32}" \
bash "${SCRIPT_DIR}/run_all_test_pipeline.sh"
