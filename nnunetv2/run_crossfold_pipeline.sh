#!/bin/bash
# ------------------------------------------------------------------------
# nnU-Net v2 cross-fold training + validation aggregation pipeline.
# Use this when there is no official independent test split. The final score
# is computed from fold validation predictions.
# ------------------------------------------------------------------------

set -e

# ================== 1. Dataset/model configuration ==================
DATASET_ID="518"

PLANS_NAME="nnUNetPlans_segmamba_bhsd_aniso"
CONFIG_NAME="segmamba_bhsd_aniso"
TRAINER_NAME="nnUNetTrainerSegMambaUIBHSDAniso"

# Train these folds. Keep 0 1 2 3 4 for full cross-validation.
FOLDS="0"
# ====================================================================


# ================== 2. GPU/resource configuration ==================
GPU_DEVICES="0,1,2,3"

# Total batch size used by nnU-Net. For DDP it must be >= number of GPUs.
TRAIN_BATCH_SIZE=32

NUM_THREADS=16
# ====================================================================


# ================== 3. nnU-Net path configuration ==================
RAW_BASE_DIR="${nnUNet_raw:-/home/wjx/CodeData/data/nnUNetData/nnUNet_raw}"
PREPROCESSED_BASE_DIR="${nnUNet_preprocessed:-/home/wjx/CodeData/data/nnUNetData/nnUNet_preprocessed}"
RESULTS_BASE_DIR="${nnUNet_results:-/home/wjx/CodeData/code/nnUNet/nnUNet_results}"
# ====================================================================


# 4. Resolve dataset name from ID.
if [ ! -d "$RAW_BASE_DIR" ]; then
    echo "ERROR: nnUNet_raw root not found: $RAW_BASE_DIR"
    exit 1
fi

DETECTED_NAME=$(basename "$(ls -d "${RAW_BASE_DIR}/Dataset${DATASET_ID}_"* 2>/dev/null | head -n 1)" 2>/dev/null || echo "")

if [ -z "$DETECTED_NAME" ]; then
    echo "ERROR: No Dataset${DATASET_ID}_* folder found in $RAW_BASE_DIR"
    exit 1
fi

DATASET_NAME="$DETECTED_NAME"

MODEL_DIR="${RESULTS_BASE_DIR}/${DATASET_NAME}/${TRAINER_NAME}__${PLANS_NAME}__${CONFIG_NAME}"
RAW_IMAGES="${RAW_BASE_DIR}/${DATASET_NAME}/imagesTr"
RAW_LABELS="${RAW_BASE_DIR}/${DATASET_NAME}/labelsTr"
PLANS_JSON="${PREPROCESSED_BASE_DIR}/${DATASET_NAME}/${PLANS_NAME}.json"
CV_DIR="${MODEL_DIR}/crossval_results_folds_0_1_2_3_4"


# 5. Update plans batch size before training.
echo "====================================================================================="
echo "Checking nnU-Net plans batch size"
echo "DATASET_NAME      : ${DATASET_NAME}"
echo "PLANS_JSON        : ${PLANS_JSON}"
echo "CONFIG_NAME       : ${CONFIG_NAME}"
echo "TRAIN_BATCH_SIZE  : ${TRAIN_BATCH_SIZE}"
echo "====================================================================================="

if [ ! -f "$PLANS_JSON" ]; then
    echo "ERROR: plans file not found: $PLANS_JSON"
    exit 1
fi

BACKUP_PLANS_JSON="${PLANS_JSON}.before_crossfold_batchsize_edit.bak"
if [ ! -f "$BACKUP_PLANS_JSON" ]; then
    cp "$PLANS_JSON" "$BACKUP_PLANS_JSON"
    echo "Backed up plans file to: $BACKUP_PLANS_JSON"
else
    echo "Backup already exists: $BACKUP_PLANS_JSON"
fi

python3 - <<PY
import json
from pathlib import Path

plans_json = Path("${PLANS_JSON}")
config_name = "${CONFIG_NAME}"
batch_size = int("${TRAIN_BATCH_SIZE}")

plans = json.loads(plans_json.read_text())
if config_name not in plans.get("configurations", {}):
    raise RuntimeError(
        f"Config {config_name!r} not found in {plans_json}. "
        f"Available: {list(plans.get('configurations', {}).keys())}"
    )

old_bs = plans["configurations"][config_name].get("batch_size")
plans["configurations"][config_name]["batch_size"] = batch_size
plans_json.write_text(json.dumps(plans, indent=4))
print(f"batch_size: {old_bs} -> {batch_size}")
PY


# 6. GPU count and sanity checks.
if [[ "$GPU_DEVICES" == *","* ]]; then
    NUM_GPUS=$(echo "$GPU_DEVICES" | tr -cd ',' | wc -c)
    NUM_GPUS=$((NUM_GPUS + 1))
else
    NUM_GPUS=1
fi

if [ "$TRAIN_BATCH_SIZE" -lt "$NUM_GPUS" ]; then
    echo "ERROR: TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} < NUM_GPUS=${NUM_GPUS}"
    exit 1
fi

if [ ! -d "$RAW_IMAGES" ]; then
    echo "ERROR: missing imagesTr folder: $RAW_IMAGES"
    exit 1
fi

if [ ! -d "$RAW_LABELS" ]; then
    echo "ERROR: missing labelsTr folder: $RAW_LABELS"
    exit 1
fi


echo "====================================================================================="
echo "Starting cross-fold pipeline"
echo "DATASET_ID        : ${DATASET_ID}"
echo "DATASET_NAME      : ${DATASET_NAME}"
echo "TRAINER_NAME      : ${TRAINER_NAME}"
echo "PLANS_NAME        : ${PLANS_NAME}"
echo "CONFIG_NAME       : ${CONFIG_NAME}"
echo "FOLDS             : ${FOLDS}"
echo "GPU_DEVICES       : ${GPU_DEVICES}"
echo "NUM_GPUS          : ${NUM_GPUS}"
echo "TRAIN_BATCH_SIZE  : ${TRAIN_BATCH_SIZE}"
echo "MODEL_DIR         : ${MODEL_DIR}"
echo "RAW_IMAGES        : ${RAW_IMAGES}"
echo "RAW_LABELS        : ${RAW_LABELS}"
echo "====================================================================================="


# STEP 1: Train requested folds.
# for fold in ${FOLDS}; do
#     echo "-----------------------------------------------------------------"
#     echo "[STEP 1] Training fold ${fold}"
#     echo "-----------------------------------------------------------------"

#     if [ "$NUM_GPUS" -gt 1 ]; then
#         CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" nnUNetv2_train \
#           "${DATASET_NAME}" \
#           "${CONFIG_NAME}" \
#           "${fold}" \
#           -tr "${TRAINER_NAME}" \
#           -num_gpus "${NUM_GPUS}" \
#           -p "${PLANS_NAME}"
#     else
#         CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" nnUNetv2_train \
#           "${DATASET_NAME}" \
#           "${CONFIG_NAME}" \
#           "${fold}" \
#           -tr "${TRAINER_NAME}" \
#           -p "${PLANS_NAME}"
#     fi
# done


# STEP 2: Accumulate cross-validation predictions and determine postprocessing.
# nnUNetv2_find_best_configuration expects all five folds for the standard CV
# folder. If FOLDS is not all five folds, use fold_N/validation/summary.json.
if [ "${FOLDS}" = "0 1 2 3 4" ]; then
    echo "-----------------------------------------------------------------"
    echo "[STEP 2] Accumulating CV results and determining postprocessing"
    echo "-----------------------------------------------------------------"
    nnUNetv2_find_best_configuration \
      "${DATASET_ID}" \
      -c "${CONFIG_NAME}" \
      -tr "${TRAINER_NAME}" \
      -p "${PLANS_NAME}" \
      -np "${NUM_THREADS}"

    echo "-----------------------------------------------------------------"
    echo "[STEP 3] Final cross-validation results"
    echo "-----------------------------------------------------------------"
    if [ -f "${CV_DIR}/postprocessed/summary.json" ]; then
        echo "Final postprocessed CV summary:"
        echo "${CV_DIR}/postprocessed/summary.json"
    elif [ -f "${CV_DIR}/summary.json" ]; then
        echo "Final raw CV summary:"
        echo "${CV_DIR}/summary.json"
    else
        echo "ERROR: expected CV summary not found under ${CV_DIR}"
        exit 1
    fi
else
    echo "-----------------------------------------------------------------"
    echo "[STEP 2] Non-5-fold run complete"
    echo "-----------------------------------------------------------------"
    echo "You did not run all five folds. Use each fold validation summary directly:"
    for fold in ${FOLDS}; do
        summary="${MODEL_DIR}/fold_${fold}/validation/summary.json"
        if [ -f "$summary" ]; then
            echo "$summary"
        else
            echo "WARNING: missing $summary"
        fi
    done
fi

echo "====================================================================================="
echo "Cross-fold pipeline complete."
echo "====================================================================================="
