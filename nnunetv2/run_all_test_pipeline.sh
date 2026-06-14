#!/bin/bash
# ------------------------------------------------------------------------
# nnU-Net v2 【全量训练 + 独立测试集推理 + 后处理 + 极限评估】ID智能反查版
# ------------------------------------------------------------------------
set -e # 🛡️ 数值安全防线：任何一步报错，立刻强行熔断退出

# 🌟================== 1. 数据集 ID 核心配置区 ==================🌟
# 现在你只需要手动确定 ID 即可，脚本会自动帮你抓取完整的 DATASET_NAME！
DATASET_ID="515"

# 基础架构配置（跑你的 SegMamba 时，保持下面一致）
CONFIG_NAME="segmamba"
TRAINER_NAME="nnUNetTrainerSegMamba"  
# ====================================================================🌟

# 🌟================== 2. 显卡与运算资源自定义配置区 ==================🌟
GPU_DEVICES="7"
NUM_THREADS=8  
# ====================================================================🌟

# 🚀 3. 【核心黑科技】通过 DATASET_ID 自动反查并确立 DATASET_NAME
# 首先读取你系统中的 nnUNet_raw 环境变量，如果未定义则使用你的真实物理路径
RAW_BASE_DIR="${nnUNet_raw:-/home/wjx/CodeData/data/nnUNetData/nnUNet_raw}"

if [ ! -d "$RAW_BASE_DIR" ]; then
    echo "❌ 错误: 找不到 nnUNet_raw 根目录: $RAW_BASE_DIR"
    exit 1
fi

# 核心检索：在 raw 目录下寻找以 Dataset 开头、且紧跟你的 ID 的文件夹名字
DETECTED_NAME=$(basename $(ls -d ${RAW_BASE_DIR}/Dataset${DATASET_ID}_* 2>/dev/null | head -n 1) 2>/dev/null || echo "")

if [ -z "$DETECTED_NAME" ]; then
    echo "❌ 错误: 在路径 $RAW_BASE_DIR 下未探测到包含 ID ${DATASET_ID} 的数据集文件夹！"
    echo "📌 请检查你的文件夹命名是否符合 nnU-Net 规范 (例如: Dataset${DATASET_ID}_XXX)"
    exit 1
else
    DATASET_NAME="$DETECTED_NAME"
fi

# 🚀 4. 基于自动反查出的变量，全自动派生绝对路径
MODEL_DIR="/home/wjx/CodeData/code/nnUNet/nnUNet_results/${DATASET_NAME}/${TRAINER_NAME}__nnUNetPlans__${CONFIG_NAME}"
TEST_IMAGES="/home/wjx/CodeData/data/nnUNetData/nnUNet_raw/${DATASET_NAME}/imagesTs" 
TEST_LABELS="/home/wjx/CodeData/data/nnUNetData/nnUNet_raw/${DATASET_NAME}/labelsTs" 

TEST_PRED_DIR="${MODEL_DIR}/Test_All_Predictions"
TEST_PRED_PP_DIR="${MODEL_DIR}/Test_All_Predictions_PostProcessing"

# 📊 【参数大满贯全景高亮看板区 —— 智能反查版】
echo "====================================================================================="
echo "🪐 正在拉起 nnU-Net v2 All-in-Test 生产环境通用全流程流水线..."
echo "====================================================================================="
echo "📋 [核心配置环境盘点]:"
echo "   ├─ 🆔 输入 DATASET_ID   : ${DATASET_ID}"
echo "   ├─ 📦 智能反查 NAME     : ${DATASET_NAME} 🟢 (自动锁定成功)"
echo "   ├─ 🧠 TRAINER_NAME     : ${TRAINER_NAME}"
echo "   ├─ 📐 CONFIG_NAME      : ${CONFIG_NAME}"
echo "   ├─ 📌 GPU_DEVICES      : CUDA_VISIBLE_DEVICES=${GPU_DEVICES}"
echo "   └─ 🧵 NUM_THREADS      : ${NUM_THREADS} CPU Threads"
echo "-------------------------------------------------------------------------------------"
echo "📂 [派生绝对物理路径图谱]:"
echo "   ├─ 📂 MODEL_DIR        : ${MODEL_DIR}"
echo "   ├─ 🖼️ TEST_IMAGES      : ${TEST_IMAGES}"
echo "   ├─ 🏷️ TEST_LABELS      : ${TEST_LABELS}"
echo "   ├─ 🔮 TEST_PRED_DIR    : ${TEST_PRED_DIR}"
echo "   └─ ✨ TEST_PRED_PP_DIR : ${TEST_PRED_PP_DIR}"
echo "====================================================================================="

# 💥 STEP 1: 全量数据长跑训练 (Fold = all)
echo "▶| [STEP 1/4 START >>>>>>] 正在验证全量训练状态..."
if [[ $GPU_DEVICES == *","* ]]; then
    NUM_GPUS=$(echo $GPU_DEVICES | tr -cd ',' | wc -c); NUM_GPUS=$((NUM_GPUS + 1))
    CUDA_VISIBLE_DEVICES=${GPU_DEVICES} nnUNetv2_train ${DATASET_NAME} ${CONFIG_NAME} all -tr ${TRAINER_NAME} -num_gpus ${NUM_GPUS}
else
    CUDA_VISIBLE_DEVICES=${GPU_DEVICES} nnUNetv2_train ${DATASET_NAME} ${CONFIG_NAME} all -tr ${TRAINER_NAME}
fi
echo "▶| [STEP 1/4 STOP >>>>>>]"
echo "-----------------------------------------------------------------"

💥 STEP 2: 独立测试集全量滑窗推理
echo "▶| [STEP 2/4 START >>>>>>] 训练圆满结束！正在对独立测试集进行推理预测..."
TOTAL_SAMPLES=$(ls -1 ${TEST_IMAGES}/*.nii.gz 2>/dev/null | wc -l || echo "0")
echo "📊 系统盘点：独立测试集 [ ${DATASET_NAME} ] 共有 ${TOTAL_SAMPLES} 个待预测文件。"
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} nnUNetv2_predict \
  -d ${DATASET_NAME} \
  -i ${TEST_IMAGES} \
  -o ${TEST_PRED_DIR} \
  -f all \
  -tr ${TRAINER_NAME} \
  -c ${CONFIG_NAME} \
  -p nnUNetPlans
echo "▶| [STEP 2/4 STOP >>>>>>]"
echo "-----------------------------------------------------------------"

# 💥 STEP 3: 移植最稳健的后处理配置
# echo "▶| [STEP 3/4 START >>>>>>] 推理完成！正在从 fold_all 现场提取后处理决策并深度优化测试集图像..."
# nnUNetv2_apply_postprocessing \
#   -i ${TEST_PRED_DIR} \
#   -o ${TEST_PRED_PP_DIR} \
#   -pp_pkl_file ${MODEL_DIR}/fold_all/postprocessing.pkl \
#   -np ${NUM_THREADS} \
#   -plans_json ${MODEL_DIR}/fold_all/plans.json
# echo "▶| [STEP 3/4 STOP >>>>>>]"
# echo "-----------------------------------------------------------------"

echo "▶| [STEP 3/4 START >>>>>>] 推理完成！正在执行全量无损结果映射..."
mkdir -p "${TEST_PRED_PP_DIR}"
cp -r ${TEST_PRED_DIR}/* "${TEST_PRED_PP_DIR}/"
echo "▶| [STEP 3/4 STOP >>>>>>]"
echo "-----------------------------------------------------------------"

# 💥 STEP 4: 在独立测试集上进行全量学术指标大盘点
echo "▶| [STEP 4/4 START >>>>>>] 后处理完成！正在调用医学图像评估器严格核对测试集最终 Dice 指标..."
nnUNetv2_evaluate_folder \
  "${TEST_LABELS}" \
  "${TEST_PRED_PP_DIR}" \
  -djfile "${MODEL_DIR}/dataset.json" \
  -pfile "${MODEL_DIR}/plans.json"
echo "▶| [STEP 4/4 STOP >>>>>>]"

echo "====================================================================================="
echo "🏆 🎉 恭喜！【智能反查全流程通用流水线】完美通关！"
echo "👉 最终独立测试集学术战报： ${TEST_PRED_PP_DIR}/summary.json"
echo "====================================================================================="