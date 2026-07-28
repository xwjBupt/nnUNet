#!/bin/bash
# ------------------------------------------------------------------------
# nnU-Net v2 【全量训练 + 独立测试集推理 + 后处理 + 极限评估】ID智能反查版
# 支持：训练前自动修改 plans.json 中的 batch_size
# ------------------------------------------------------------------------

set -e  # 🛡️ 数值安全防线：任何一步报错，立刻强行熔断退出

# 🌟================== 1. 数据集 ID 核心配置区 ==================🌟
# 现在你只需要手动确定 ID 即可，脚本会自动帮你抓取完整的 DATASET_NAME
DATASET_ID="515"

# 基础架构配置
# 跑你的 SegMamba 时，保持下面一致
PLANS_NAME="nnUNetPlans_segmamba_ui"
CONFIG_NAME="segmamba_uig_lite_128x96x96"
TRAINER_NAME="nnUNetTrainerSegMambaUIGStable"
# ====================================================================🌟


# 🌟================== 2. 显卡与运算资源自定义配置区 ==================🌟
GPU_DEVICES="0,1,2,3,4,5,6,7"  # 你想用的 GPU 卡号，逗号分隔

# ✅ 这里修改训练 batch size
# 注意：
# 1. 这是 nnU-Net 训练使用的“总 batch size”
# 2. 如果你用 8 张卡，TRAIN_BATCH_SIZE=8 通常相当于每卡 batch size≈1
# 3. 如果想每卡 batch size≈2，8 张卡时应设置为 16
# 4. 如果想每卡 batch size≈4，8 张卡时应设置为 32
TRAIN_BATCH_SIZE=16

NUM_THREADS=32
# ====================================================================🌟


# 🌟================== 3. nnU-Net 路径配置区 ==================🌟
# 如果系统环境变量中已经设置 nnUNet_raw / nnUNet_preprocessed / nnUNet_results，
# 则优先使用环境变量；否则使用你当前脚本中的默认物理路径。

RAW_BASE_DIR="${nnUNet_raw:-/home/wjx/CodeData/data/nnUNetData/nnUNet_raw}"
PREPROCESSED_BASE_DIR="${nnUNet_preprocessed:-/home/wjx/CodeData/data/nnUNetData/nnUNet_preprocessed}"
RESULTS_BASE_DIR="${nnUNet_results:-/home/wjx/CodeData/code/nnUNet/nnUNet_results}"
# ====================================================================🌟


# 🚀 4. 通过 DATASET_ID 自动反查并确立 DATASET_NAME
if [ ! -d "$RAW_BASE_DIR" ]; then
    echo "❌ 错误: 找不到 nnUNet_raw 根目录: $RAW_BASE_DIR"
    exit 1
fi

# 核心检索：在 raw 目录下寻找以 Dataset 开头、且紧跟你的 ID 的文件夹名字
DETECTED_NAME=$(basename "$(ls -d "${RAW_BASE_DIR}/Dataset${DATASET_ID}_"* 2>/dev/null | head -n 1)" 2>/dev/null || echo "")

if [ -z "$DETECTED_NAME" ]; then
    echo "❌ 错误: 在路径 $RAW_BASE_DIR 下未探测到包含 ID ${DATASET_ID} 的数据集文件夹！"
    echo "📌 请检查你的文件夹命名是否符合 nnU-Net 规范，例如: Dataset${DATASET_ID}_XXX"
    exit 1
else
    DATASET_NAME="$DETECTED_NAME"
fi


# 🚀 5. 基于自动反查出的变量，全自动派生绝对路径
MODEL_DIR="${RESULTS_BASE_DIR}/${DATASET_NAME}/${TRAINER_NAME}__${PLANS_NAME}__${CONFIG_NAME}"

TEST_IMAGES="${RAW_BASE_DIR}/${DATASET_NAME}/imagesTs"
TEST_LABELS="${RAW_BASE_DIR}/${DATASET_NAME}/labelsTs"

TEST_PRED_DIR="${MODEL_DIR}/Test_All_Predictions"
TEST_PRED_PP_DIR="${MODEL_DIR}/Test_All_Predictions_PostProcessing"

PLANS_JSON="${PREPROCESSED_BASE_DIR}/${DATASET_NAME}/${PLANS_NAME}.json"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_COPY_PATH="${MODEL_DIR}/$(basename "$SCRIPT_PATH")"

save_pipeline_script() {
    local exit_code=$?

    if mkdir -p "$MODEL_DIR" && cp "$SCRIPT_PATH" "$SCRIPT_COPY_PATH"; then
        echo "📄 已保存当前流水线脚本副本到: $SCRIPT_COPY_PATH"
    else
        echo "⚠️ 警告: 保存当前流水线脚本副本失败: $SCRIPT_COPY_PATH"
    fi

    return "$exit_code"
}

trap save_pipeline_script EXIT


# 将一个 nnU-Net summary.json 追加到结果看板；同一实验重复运行时更新原记录。
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
    raise FileNotFoundError(f"评估结果不存在: {summary_path}")

try:
    relative_path = model_dir.relative_to(results_dir / dataset).as_posix()
except ValueError as exc:
    raise RuntimeError(
        f"模型目录 {model_dir} 不在数据集结果目录 {results_dir / dataset} 下"
    ) from exc

with summary_path.open("r", encoding="utf-8") as f:
    foreground_mean = json.load(f)["foreground_mean"]

missing_metrics = [name for name in metric_names if name not in foreground_mean]
if missing_metrics:
    raise RuntimeError(
        f"{summary_path} 的 foreground_mean 缺少指标: {missing_metrics}"
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
                    f"{board_path} 列格式不匹配: {reader.fieldnames}; 期望: {fieldnames}"
                )
            rows = list(reader)

    key = (dataset, relative_path, fold)
    old_count = len(rows)
    rows = [
        row for row in rows
        if (row["dataset"], row["relative_path"], row["fold"]) != key
    ]
    action = "更新" if len(rows) != old_count else "追加"
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

print(f"📊 已{action}结果看板: {dataset}, {relative_path}, {fold}")
PY
}


# 🚀 6. 训练前自动修改 plans.json 中的 batch_size
echo "====================================================================================="
echo "🛠️ 正在检查并修改 nnU-Net plans.json 中的 batch_size..."
echo "====================================================================================="
echo "📂 PREPROCESSED_BASE_DIR : ${PREPROCESSED_BASE_DIR}"
echo "📄 PLANS_JSON            : ${PLANS_JSON}"
echo "📐 CONFIG_NAME           : ${CONFIG_NAME}"
echo "🧩 TARGET BATCH_SIZE     : ${TRAIN_BATCH_SIZE}"
echo "-------------------------------------------------------------------------------------"

if [ ! -f "$PLANS_JSON" ]; then
    echo "❌ 错误: 找不到 plans 文件: $PLANS_JSON"
    echo "📌 可能原因："
    echo "   1. 你还没有运行 nnUNetv2_plan_and_preprocess"
    echo "   2. PLANS_NAME 写错了"
    echo "   3. nnUNet_preprocessed 路径不对"
    exit 1
fi

# 自动备份 plans.json，只保留第一次备份，避免反复覆盖原始配置
BACKUP_PLANS_JSON="${PLANS_JSON}.before_batchsize_edit.bak"
if [ ! -f "$BACKUP_PLANS_JSON" ]; then
    cp "$PLANS_JSON" "$BACKUP_PLANS_JSON"
    echo "✅ 已备份原始 plans 文件到: ${BACKUP_PLANS_JSON}"
else
    echo "ℹ️ 已存在备份文件，不重复覆盖: ${BACKUP_PLANS_JSON}"
fi

python3 - <<PY
import json
from pathlib import Path

plans_json = Path("${PLANS_JSON}")
config_name = "${CONFIG_NAME}"
batch_size = int("${TRAIN_BATCH_SIZE}")

with plans_json.open("r", encoding="utf-8") as f:
    plans = json.load(f)

if "configurations" not in plans:
    raise RuntimeError("❌ plans.json 中没有 configurations 字段，无法修改 batch_size")

if config_name not in plans["configurations"]:
    available = list(plans["configurations"].keys())
    raise RuntimeError(
        f"❌ plans.json 中找不到配置 '{config_name}'。"
        f" 当前可用配置为: {available}"
    )

old_bs = plans["configurations"][config_name].get("batch_size", None)
plans["configurations"][config_name]["batch_size"] = batch_size

with plans_json.open("w", encoding="utf-8") as f:
    json.dump(plans, f, indent=4)

print(f"✅ batch_size 修改完成: {old_bs} -> {batch_size}")
PY

echo "====================================================================================="


# 🚀 7. 计算 GPU 数量
if [[ "$GPU_DEVICES" == *","* ]]; then
    NUM_GPUS=$(echo "$GPU_DEVICES" | tr -cd ',' | wc -c)
    NUM_GPUS=$((NUM_GPUS + 1))
else
    NUM_GPUS=1
fi

if [ "$TRAIN_BATCH_SIZE" -lt "$NUM_GPUS" ]; then
    echo "❌ 错误: TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} 小于 GPU 数量 NUM_GPUS=${NUM_GPUS}"
    echo "📌 nnU-Net 多卡训练时，总 batch size 至少应 >= GPU 数量。"
    echo "📌 例如 8 张卡时，TRAIN_BATCH_SIZE 至少设为 8。"
    exit 1
fi


# 📊 参数大满贯全景高亮看板区
echo "====================================================================================="
echo "🪐 正在拉起 nnU-Net v2 All-in-Test 生产环境通用全流程流水线..."
echo "====================================================================================="
echo "📋 [核心配置环境盘点]:"
echo "   ├─ 🆔 输入 DATASET_ID      : ${DATASET_ID}"
echo "   ├─ 📦 智能反查 NAME        : ${DATASET_NAME} 🟢 自动锁定成功"
echo "   ├─ 🧠 TRAINER_NAME        : ${TRAINER_NAME}"
echo "   ├─ 📐 CONFIG_NAME         : ${CONFIG_NAME}"
echo "   ├─ 📐 PLANS_NAME          : ${PLANS_NAME}"
echo "   ├─ 📌 GPU_DEVICES         : CUDA_VISIBLE_DEVICES=${GPU_DEVICES}"
echo "   ├─ 🧮 NUM_GPUS            : ${NUM_GPUS}"
echo "   ├─ 📦 TRAIN_BATCH_SIZE    : ${TRAIN_BATCH_SIZE}"
echo "   └─ 🧵 NUM_THREADS         : ${NUM_THREADS} CPU Threads"
echo "-------------------------------------------------------------------------------------"
echo "📂 [派生绝对物理路径图谱]:"
echo "   ├─ 📂 RAW_BASE_DIR        : ${RAW_BASE_DIR}"
echo "   ├─ 📂 PREPROCESSED_DIR    : ${PREPROCESSED_BASE_DIR}"
echo "   ├─ 📂 RESULTS_BASE_DIR    : ${RESULTS_BASE_DIR}"
echo "   ├─ 📄 PLANS_JSON          : ${PLANS_JSON}"
echo "   ├─ 📂 MODEL_DIR           : ${MODEL_DIR}"
echo "   ├─ 🖼️ TEST_IMAGES         : ${TEST_IMAGES}"
echo "   ├─ 🏷️ TEST_LABELS         : ${TEST_LABELS}"
echo "   ├─ 🔮 TEST_PRED_DIR       : ${TEST_PRED_DIR}"
echo "   └─ ✨ TEST_PRED_PP_DIR    : ${TEST_PRED_PP_DIR}"
echo "====================================================================================="


# 💥 STEP 1: 全量数据长跑训练，Fold = all
echo "▶| [STEP 1/4 START >>>>>>] 正在启动全量训练..."
echo "⚠️ 提醒：如果该 MODEL_DIR 下已经存在旧 checkpoint，batch_size 修改不会自动改变旧 checkpoint 的历史训练过程。"
echo "⚠️ 如果你想完全用新 batch_size 从头训练，建议先手动确认是否需要清理旧的 fold_all 结果目录。"

if [ "$NUM_GPUS" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" nnUNetv2_train \
      "${DATASET_NAME}" \
      "${CONFIG_NAME}" \
      all \
      -tr "${TRAINER_NAME}" \
      -num_gpus "${NUM_GPUS}" \
      -p "${PLANS_NAME}"
else
    CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" nnUNetv2_train \
      "${DATASET_NAME}" \
      "${CONFIG_NAME}" \
      all \
      -tr "${TRAINER_NAME}" \
      -p "${PLANS_NAME}"
fi

update_resultsboard \
  "${MODEL_DIR}/fold_all/validation/summary.json" \
  "fold_all/validation"

echo "▶| [STEP 1/4 STOP >>>>>>]"
echo "-----------------------------------------------------------------"


# 💥 STEP 2: 独立测试集全量滑窗推理
echo "▶| [STEP 2/4 START >>>>>>] 训练结束！正在对独立测试集进行推理预测..."

if [ ! -d "$TEST_IMAGES" ]; then
    echo "❌ 错误: 找不到测试图像目录: $TEST_IMAGES"
    exit 1
fi

TOTAL_SAMPLES=$(ls -1 "${TEST_IMAGES}"/*.nii.gz 2>/dev/null | wc -l || echo "0")
echo "📊 系统盘点：独立测试集 [ ${DATASET_NAME} ] 共有 ${TOTAL_SAMPLES} 个待预测文件。"

if [ "$TOTAL_SAMPLES" -eq 0 ]; then
    echo "❌ 错误: ${TEST_IMAGES} 下没有找到 .nii.gz 测试图像。"
    exit 1
fi

mkdir -p "${TEST_PRED_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" nnUNetv2_predict \
  -d "${DATASET_NAME}" \
  -i "${TEST_IMAGES}" \
  -o "${TEST_PRED_DIR}" \
  -f all \
  -tr "${TRAINER_NAME}" \
  -c "${CONFIG_NAME}" \
  -p "${PLANS_NAME}"

echo "▶| [STEP 2/4 STOP >>>>>>]"
echo "-----------------------------------------------------------------"


# 💥 STEP 3: 后处理
# 如果你有真实 postprocessing.pkl，可以启用下面这段正式后处理。
# 当前默认采用“无损复制预测结果”的方式，保持你原脚本逻辑。
#
# echo "▶| [STEP 3/4 START >>>>>>] 推理完成！正在从 fold_all 提取后处理决策并优化测试集图像..."
# nnUNetv2_apply_postprocessing \
#   -i "${TEST_PRED_DIR}" \
#   -o "${TEST_PRED_PP_DIR}" \
#   -pp_pkl_file "${MODEL_DIR}/fold_all/postprocessing.pkl" \
#   -np "${NUM_THREADS}" \
#   -plans_json "${MODEL_DIR}/fold_all/plans.json"
# echo "▶| [STEP 3/4 STOP >>>>>>]"
# echo "-----------------------------------------------------------------"

echo "▶| [STEP 3/4 START >>>>>>] 推理完成！正在执行全量无损结果映射..."
mkdir -p "${TEST_PRED_PP_DIR}"
cp -r "${TEST_PRED_DIR}/"* "${TEST_PRED_PP_DIR}/"
echo "▶| [STEP 3/4 STOP >>>>>>]"
echo "-----------------------------------------------------------------"


# 💥 STEP 4: 在独立测试集上进行全量学术指标大盘点
echo "▶| [STEP 4/4 START >>>>>>] 后处理完成！正在调用医学图像评估器严格核对测试集最终 Dice 指标..."

if [ ! -d "$TEST_LABELS" ]; then
    echo "❌ 错误: 找不到测试标签目录: $TEST_LABELS"
    exit 1
fi

if [ ! -f "${MODEL_DIR}/dataset.json" ]; then
    echo "❌ 错误: 找不到 ${MODEL_DIR}/dataset.json"
    echo "📌 请确认训练是否成功完成，或 MODEL_DIR 是否正确。"
    exit 1
fi

if [ ! -f "${MODEL_DIR}/plans.json" ]; then
    echo "❌ 错误: 找不到 ${MODEL_DIR}/plans.json"
    echo "📌 请确认训练是否成功完成，或 MODEL_DIR 是否正确。"
    exit 1
fi

nnUNetv2_evaluate_folder \
  "${TEST_LABELS}" \
  "${TEST_PRED_PP_DIR}" \
  -djfile "${MODEL_DIR}/dataset.json" \
  -pfile "${MODEL_DIR}/plans.json"

update_resultsboard \
  "${TEST_PRED_PP_DIR}/summary.json" \
  "Test_All_Predictions_PostProcessing"

echo "▶| [STEP 4/4 STOP >>>>>>]"


echo "====================================================================================="
echo "🏆 🎉 恭喜！【智能反查全流程通用流水线】完美通关！"
echo "👉 最终独立测试集学术战报： ${TEST_PRED_PP_DIR}/summary.json"
echo "====================================================================================="
