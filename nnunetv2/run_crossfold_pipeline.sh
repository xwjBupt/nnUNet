#!/bin/bash
# ------------------------------------------------------------------------
# nnU-Net v2 cross-fold training + validation aggregation pipeline.
# Use this when there is no official independent test split. The final score
# is computed from fold validation predictions.
# ------------------------------------------------------------------------

set -e

# ================== 1. Dataset/model configuration ==================
DATASET_ID="517"

PLANS_NAME="nnUNetPlans_segmamba_ui"
CONFIG_NAME="segmamba_uig_dec2_logit_boundary_hierarchy_core_exterior_masked_foreground_sample_dice_128x96x96"
TRAINER_NAME="nnUNetTrainerSegMambaUIGStableHierarchyCoreExteriorMaskedForegroundSampleDice"

# Train these folds. Keep 0 1 2 3 4 for full cross-validation.
FOLDS="0"
# ====================================================================


# ================== 2. GPU/resource configuration ==================
GPU_DEVICES="0,1,2,3,4,5,6,7"

# Total batch size used by nnU-Net. For DDP it must be >= number of GPUs.
TRAIN_BATCH_SIZE=24

NUM_THREADS=16
# ====================================================================


# ================== 3. nnU-Net path configuration ==================
RAW_BASE_DIR="${nnUNet_raw:-/home/wjx/CodeData/data/nnUNetData/nnUNet_raw}"
PREPROCESSED_BASE_DIR="${nnUNet_preprocessed:-/home/wjx/CodeData/data/nnUNetData/nnUNet_preprocessed}"
RESULTS_BASE_DIR="${nnUNet_results:-/home/wjx/CodeData/code/nnUNet/nnUNet_results}"
# ====================================================================

NNUNET_ENV_BIN="${NNUNET_ENV_BIN:-/home/wjx/miniconda3/envs/nnunet_seg/bin}"
export PATH="${NNUNET_ENV_BIN}:${PATH}"
export nnUNet_raw="${RAW_BASE_DIR}"
export nnUNet_preprocessed="${PREPROCESSED_BASE_DIR}"
export nnUNet_results="${RESULTS_BASE_DIR}"
export nnUNet_compile=false


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
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_COPY_PATH="${MODEL_DIR}/$(basename "$SCRIPT_PATH")"

save_pipeline_script() {
    local exit_code=$?

    if mkdir -p "$MODEL_DIR" && cp "$SCRIPT_PATH" "$SCRIPT_COPY_PATH"; then
        echo "Saved pipeline script copy to: $SCRIPT_COPY_PATH"
    else
        echo "WARNING: failed to save pipeline script copy to: $SCRIPT_COPY_PATH"
    fi

    return "$exit_code"
}

trap save_pipeline_script EXIT


# Append one nnU-Net summary.json to the results board. Re-running the same
# experiment updates its existing row instead of creating a duplicate.
update_resultsboard() {
    local summary_path="$1"
    local fold_value="$2"

    python3 - \
      "$RESULTS_BASE_DIR" \
      "$DATASET_NAME" \
      "$MODEL_DIR" \
      "$summary_path" \
      "$fold_value" <<'PY'
import csv
import fcntl
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

results_dir = Path(sys.argv[1]).resolve()
dataset = sys.argv[2]
model_dir = Path(sys.argv[3]).resolve()
summary_path = Path(sys.argv[4]).resolve()
fold = sys.argv[5]
board_path = results_dir / "resultsboard.csv"
lock_path = results_dir / ".resultsboard.csv.lock"
metric_names = ["Dice", "FN", "FP", "IoU", "TN", "TP", "n_pred", "n_ref"]
fieldnames = ["dataset", "relative_path", "fold", *metric_names]

if not summary_path.is_file():
    raise FileNotFoundError(f"Summary file not found: {summary_path}")

try:
    relative_path = model_dir.relative_to(results_dir / dataset).as_posix()
except ValueError as exc:
    raise RuntimeError(
        f"Model directory {model_dir} is not under {results_dir / dataset}"
    ) from exc

with summary_path.open("r", encoding="utf-8") as f:
    foreground_mean = json.load(f)["foreground_mean"]

missing_metrics = [name for name in metric_names if name not in foreground_mean]
if missing_metrics:
    raise RuntimeError(
        f"{summary_path} foreground_mean is missing metrics: {missing_metrics}"
    )

new_row = {
    "dataset": dataset,
    "relative_path": relative_path,
    "fold": fold,
    **{name: f"{float(foreground_mean[name]):.5f}" for name in metric_names},
}

results_dir.mkdir(parents=True, exist_ok=True)
with lock_path.open("a+", encoding="utf-8") as lock_file:
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

    rows = []
    if board_path.is_file() and board_path.stat().st_size:
        with board_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != fieldnames:
                raise RuntimeError(
                    f"{board_path} columns are {reader.fieldnames}; expected {fieldnames}"
                )
            rows = list(reader)

    key = (dataset, relative_path, fold)
    old_count = len(rows)
    rows = [
        row for row in rows
        if (row["dataset"], row["relative_path"], row["fold"]) != key
    ]
    action = "Updated" if len(rows) != old_count else "Appended"
    rows.append(new_row)
    rows.sort(key=lambda row: (row["dataset"], row["relative_path"], row["fold"]))

    fd, temp_name = tempfile.mkstemp(
        prefix=".resultsboard.", suffix=".csv", dir=results_dir
    )
    try:
        if board_path.exists():
            os.chmod(temp_name, stat.S_IMODE(board_path.stat().st_mode))
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, board_path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise

print(f"{action} results board: {dataset}, {relative_path}, {fold}")
PY
}


# Print one fold's validation metrics immediately after it finishes.
print_fold_result() {
    local summary_path="$1"
    local fold_value="$2"

    python3 - \
      "$summary_path" \
      "$fold_value" \
      "$DATASET_NAME" \
      "$TRAINER_NAME" \
      "$PLANS_NAME" \
      "$CONFIG_NAME" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

summary_path = Path(sys.argv[1]).resolve()
fold = sys.argv[2]
dataset = sys.argv[3]
trainer = sys.argv[4]
plans = sys.argv[5]
configuration = sys.argv[6]

if not summary_path.is_file():
    raise FileNotFoundError(f"Fold summary not found: {summary_path}")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
foreground_mean = summary.get("foreground_mean")
cases = summary.get("metric_per_case")
if not isinstance(foreground_mean, dict) or not isinstance(cases, list):
    raise RuntimeError(f"Invalid nnU-Net summary: {summary_path}")


def foreground_case_metrics():
    for case in cases:
        for label, metrics in case.get("metrics", {}).items():
            if str(label).lower() not in {"0", "background"}:
                yield metrics


metrics = list(foreground_case_metrics())
dice = [float(item["Dice"]) for item in metrics if math.isfinite(float(item["Dice"]))]
precision = [
    float(item["TP"]) / (float(item["TP"]) + float(item["FP"]))
    for item in metrics
    if float(item["TP"]) + float(item["FP"]) > 0
]
recall = [
    float(item["TP"]) / (float(item["TP"]) + float(item["FN"]))
    for item in metrics
    if float(item["TP"]) + float(item["FN"]) > 0
]
if not dice:
    raise RuntimeError(f"No finite foreground Dice values in {summary_path}")

print()
print("=" * 100)
print(f"Fold {fold} validation result")
print("=" * 100)
print(f"Dataset      : {dataset}")
print(f"Trainer      : {trainer}")
print(f"Plans        : {plans}")
print(f"Configuration: {configuration}")
print(f"Cases        : {len(cases)}")
print("-" * 100)
print(f"Dice mean    : {float(foreground_mean['Dice']):.5f}")
print(f"Dice std     : {statistics.pstdev(dice):.5f}")
print(f"Dice median  : {statistics.median(dice):.5f}")
print(f"Dice range   : [{min(dice):.5f}, {max(dice):.5f}]")
print(f"IoU mean     : {float(foreground_mean['IoU']):.5f}")
print(f"Precision    : {statistics.fmean(precision):.5f} (macro)")
print(f"Recall       : {statistics.fmean(recall):.5f} (macro)")
print(f"TP / FP / FN : {float(foreground_mean['TP']):.2f} / "
      f"{float(foreground_mean['FP']):.2f} / {float(foreground_mean['FN']):.2f}")
print(f"Summary      : {summary_path}")
print("=" * 100)
PY
}


# Print a fold-by-fold table and pooled metrics after all requested folds finish.
print_all_fold_results() {
    python3 - \
      "$MODEL_DIR" \
      "$FOLDS" \
      "$DATASET_NAME" \
      "$TRAINER_NAME" \
      "$PLANS_NAME" \
      "$CONFIG_NAME" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

model_dir = Path(sys.argv[1]).resolve()
folds = sys.argv[2].split()
dataset = sys.argv[3]
trainer = sys.argv[4]
plans = sys.argv[5]
configuration = sys.argv[6]

rows = []
pooled_metrics = []
total_cases = 0
for fold in folds:
    summary_path = model_dir / f"fold_{fold}" / "validation" / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Fold summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    foreground_mean = summary["foreground_mean"]
    cases = summary["metric_per_case"]
    total_cases += len(cases)
    rows.append(
        (
            fold,
            len(cases),
            float(foreground_mean["Dice"]),
            float(foreground_mean["IoU"]),
            float(foreground_mean["FP"]),
            float(foreground_mean["FN"]),
        )
    )
    for case in cases:
        for label, metrics in case.get("metrics", {}).items():
            if str(label).lower() not in {"0", "background"}:
                pooled_metrics.append(metrics)

if not pooled_metrics:
    raise RuntimeError("No foreground metrics found in completed folds")

dice = [
    float(item["Dice"])
    for item in pooled_metrics
    if math.isfinite(float(item["Dice"]))
]
iou = [
    float(item["IoU"])
    for item in pooled_metrics
    if math.isfinite(float(item["IoU"]))
]
precision = [
    float(item["TP"]) / (float(item["TP"]) + float(item["FP"]))
    for item in pooled_metrics
    if float(item["TP"]) + float(item["FP"]) > 0
]
recall = [
    float(item["TP"]) / (float(item["TP"]) + float(item["FN"]))
    for item in pooled_metrics
    if float(item["TP"]) + float(item["FN"]) > 0
]

print()
print("=" * 100)
print("All completed fold results")
print("=" * 100)
print(f"Dataset      : {dataset}")
print(f"Trainer      : {trainer}")
print(f"Plans        : {plans}")
print(f"Configuration: {configuration}")
print(f"Folds        : {' '.join(folds)}")
print("-" * 100)
print(f"{'Fold':<8}{'Cases':>8}{'Dice':>14}{'IoU':>14}{'FP mean':>16}{'FN mean':>16}")
for fold, cases, fold_dice, fold_iou, fp, fn in rows:
    print(f"{fold:<8}{cases:>8d}{fold_dice:>14.5f}{fold_iou:>14.5f}{fp:>16.2f}{fn:>16.2f}")
print("-" * 100)
print(f"Pooled cases : {total_cases}")
print(f"Pooled Dice  : {statistics.fmean(dice):.5f}")
print(f"Dice std     : {statistics.pstdev(dice):.5f}")
print(f"Dice median  : {statistics.median(dice):.5f}")
print(f"Dice range   : [{min(dice):.5f}, {max(dice):.5f}]")
print(f"Pooled IoU   : {statistics.fmean(iou):.5f}")
print(f"Precision    : {statistics.fmean(precision):.5f} (macro)")
print(f"Recall       : {statistics.fmean(recall):.5f} (macro)")
print(f"Fold Dice    : {statistics.fmean(row[2] for row in rows):.5f} +/- "
      f"{statistics.pstdev(row[2] for row in rows):.5f}")
print("=" * 100)
PY
}


# Print the aggregated raw or postprocessed five-fold summary.
print_final_cv_result() {
    local summary_path="$1"
    local result_name="$2"

    python3 - \
      "$summary_path" \
      "$result_name" \
      "$DATASET_NAME" \
      "$TRAINER_NAME" \
      "$PLANS_NAME" \
      "$CONFIG_NAME" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

summary_path = Path(sys.argv[1]).resolve()
result_name = sys.argv[2]
dataset, trainer, plans, configuration = sys.argv[3:7]
summary = json.loads(summary_path.read_text(encoding="utf-8"))
foreground_mean = summary["foreground_mean"]
cases = summary["metric_per_case"]
metrics = [
    metric
    for case in cases
    for label, metric in case.get("metrics", {}).items()
    if str(label).lower() not in {"0", "background"}
]
dice = [float(item["Dice"]) for item in metrics if math.isfinite(float(item["Dice"]))]
precision = [
    float(item["TP"]) / (float(item["TP"]) + float(item["FP"]))
    for item in metrics
    if float(item["TP"]) + float(item["FP"]) > 0
]
recall = [
    float(item["TP"]) / (float(item["TP"]) + float(item["FN"]))
    for item in metrics
    if float(item["TP"]) + float(item["FN"]) > 0
]

print()
print("=" * 100)
print(f"Final five-fold CV result ({result_name})")
print("=" * 100)
print(f"Dataset      : {dataset}")
print(f"Trainer      : {trainer}")
print(f"Plans        : {plans}")
print(f"Configuration: {configuration}")
print(f"Cases        : {len(cases)}")
print("-" * 100)
print(f"Dice mean    : {float(foreground_mean['Dice']):.5f}")
print(f"Dice std     : {statistics.pstdev(dice):.5f}")
print(f"Dice median  : {statistics.median(dice):.5f}")
print(f"IoU mean     : {float(foreground_mean['IoU']):.5f}")
print(f"Precision    : {statistics.fmean(precision):.5f} (macro)")
print(f"Recall       : {statistics.fmean(recall):.5f} (macro)")
print(f"TP / FP / FN : {float(foreground_mean['TP']):.2f} / "
      f"{float(foreground_mean['FP']):.2f} / {float(foreground_mean['FN']):.2f}")
print(f"Summary      : {summary_path}")
print("=" * 100)
PY
}


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
for fold in ${FOLDS}; do
    echo "-----------------------------------------------------------------"
    echo "[STEP 1] Training fold ${fold}"
    echo "-----------------------------------------------------------------"

    if [ "$NUM_GPUS" -gt 1 ]; then
        CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" nnUNetv2_train \
          "${DATASET_NAME}" \
          "${CONFIG_NAME}" \
          "${fold}" \
          -tr "${TRAINER_NAME}" \
          -num_gpus "${NUM_GPUS}" \
          -p "${PLANS_NAME}"
    else
        CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" nnUNetv2_train \
          "${DATASET_NAME}" \
          "${CONFIG_NAME}" \
          "${fold}" \
          -tr "${TRAINER_NAME}" \
          -p "${PLANS_NAME}"
    fi

    update_resultsboard \
      "${MODEL_DIR}/fold_${fold}/validation/summary.json" \
      "fold${fold}/validation"

    print_fold_result \
      "${MODEL_DIR}/fold_${fold}/validation/summary.json" \
      "${fold}"
done

print_all_fold_results


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

    update_resultsboard \
      "${CV_DIR}/summary.json" \
      "crossval_results_folds_0_1_2_3_4"

    if [ -f "${CV_DIR}/postprocessed/summary.json" ]; then
        update_resultsboard \
          "${CV_DIR}/postprocessed/summary.json" \
          "crossval_results_folds_0_1_2_3_4/postprocessed"
    fi

    echo "-----------------------------------------------------------------"
    echo "[STEP 3] Final cross-validation results"
    echo "-----------------------------------------------------------------"
    if [ -f "${CV_DIR}/postprocessed/summary.json" ]; then
        echo "Final postprocessed CV summary:"
        echo "${CV_DIR}/postprocessed/summary.json"
        print_final_cv_result \
          "${CV_DIR}/postprocessed/summary.json" \
          "postprocessed"
    elif [ -f "${CV_DIR}/summary.json" ]; then
        echo "Final raw CV summary:"
        echo "${CV_DIR}/summary.json"
        print_final_cv_result \
          "${CV_DIR}/summary.json" \
          "raw"
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
