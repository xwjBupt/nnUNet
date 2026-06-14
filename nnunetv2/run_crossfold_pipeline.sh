#!/bin/bash
# ------------------------------------------------------------------------
# nnU-Net v2 【五折交叉训练 + 自动寻优 + 推理融合 + 后处理 + 新版权威评估】一劳永逸自动化版
# ------------------------------------------------------------------------
set -e # 🛡️ 数值安全防线：任何一步报错，立刻强行熔断退出

# 🌟================== 1. 数据集 ID 核心配置区 ==================🌟
# 现在你只需要手动确定 ID 即可，脚本会自动帮你抓取完整的 DATASET_NAME！
DATASET_ID="515"

# 基础架构配置（如需跑你的 SegMamba 五折，请在此无缝改写变量）
CONFIG_NAME="3d_fullres"
TRAINER_NAME="nnUNetTrainer"  
# ====================================================================🌟

# 🌟================== 2. 显卡与运算资源自定义配置区 ==================🌟
# 支持单卡（如 "5"）或多卡（如 "0,1,2,3"），脚本会自动派生 DDP 并计算张量并行数
GPU_DEVICES="5"
NUM_THREADS=8  # 后处理与数据增强使用的 CPU 线程数
# ====================================================================🌟

# 🚀 3. 【核心黑科技】通过 DATASET_ID 自动反查并确立 DATASET_NAME
RAW_BASE_DIR="${nnUNet_raw:-/home/wjx/CodeData/data/nnUNetData/nnUNet_raw}"

if [ ! -d "$RAW_BASE_DIR" ]; then
    echo "❌ 错误: 找不到 nnUNet_raw 根目录: $RAW_BASE_DIR"
    exit 1
fi

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
RAW_IMAGES="/home/wjx/CodeData/data/nnUNetData/nnUNet_raw/${DATASET_NAME}/imagesTr" 
RAW_LABELS="/home/wjx/CodeData/data/nnUNetData/nnUNet_raw/${DATASET_NAME}/labelsTr" 

# 创建五折融合输出目标文件夹
ENSEMBLE_DIR="${MODEL_DIR}/Ensemble"
POSTPROCESSED_DIR="${MODEL_DIR}/Ensemble_PostProcessing"

# 📊 【全功能参数大满贯全景高亮看板区 —— 五折交叉验证自动化版】
echo "====================================================================================="
echo "🪐 正在拉起 nnU-Net v2 Cross-Fold 学术大满贯数据集通用流水线..."
echo "====================================================================================="
echo "📋 [五折核心配置环境盘点]:"
echo "   ├─ 🆔 输入 DATASET_ID   : ${DATASET_ID}"
echo "   ├─ 📦 智能反查 NAME     : ${DATASET_NAME} 🟢 (自动锁定成功)"
echo "   ├─ 🧠 TRAINER_NAME     : ${TRAINER_NAME}"
echo "   ├─ 📐 CONFIG_NAME      : ${CONFIG_NAME}"
echo "   ├─ 📌 GPU_DEVICES      : CUDA_VISIBLE_DEVICES=${GPU_DEVICES}"
echo "   ├─ 🔄 LOOP RANGE       : Fold 0 ──> Fold 4 (串行自动接力)"
echo "   └─ 🧵 NUM_THREADS      : ${NUM_THREADS} CPU Threads"
echo "-------------------------------------------------------------------------------------"
echo "📂 [五折派生绝对物理路径图谱]:"
echo "   ├─ 📂 MODEL_DIR        : ${MODEL_DIR}"
echo "   ├─ 🖼️ RAW_IMAGES       : ${RAW_IMAGES}"
echo "   ├─ 🏷️ RAW_LABELS       : ${RAW_LABELS}"
echo "   ├─ 🔮 ENSEMBLE_DIR     : ${ENSEMBLE_DIR}"
echo "   └─ ✨ POSTPROCESSED_DIR: ${POSTPROCESSED_DIR}"
echo "====================================================================================="

# 💥 STEP 1: 自动化循环训练 Fold 0 到 Fold 4
for fold in {0..4}
do
    echo "-----------------------------------------------------------------"
    echo "⚡ [STEP 1/5] 正在启动 [ ${DATASET_NAME} ] Fold ${fold} 的标准交叉训练..."
    echo "-----------------------------------------------------------------"
    
    if [[ $GPU_DEVICES == *","* ]]; then
        NUM_GPUS=$(echo $GPU_DEVICES | tr -cd ',' | wc -c); NUM_GPUS=$((NUM_GPUS + 1))
        echo "📡 检测到多卡配置，自动激活分布式 DDP 并行训练 (GPUs: ${NUM_GPUS})..."
        CUDA_VISIBLE_DEVICES=${GPU_DEVICES} nnUNetv2_train ${DATASET_NAME} ${CONFIG_NAME} ${fold} -tr ${TRAINER_NAME} -num_gpus ${NUM_GPUS}
    else
        echo "⚡ 激活单卡训练模式..."
        CUDA_VISIBLE_DEVICES=${GPU_DEVICES} nnUNetv2_train ${DATASET_NAME} ${CONFIG_NAME} ${fold} -tr ${TRAINER_NAME}
    fi
done

# 💥 STEP 2: 五折验证前向汇总并自动搜寻黄金后处理阈值
echo "▶| [STEP 2/5 START >>>>>>] 五折训练圆满结束！正在执行全折前向汇总与后处理决策寻优..."
nnUNetv2_find_best_configuration ${DATASET_ID} -c ${CONFIG_NAME} -tr ${TRAINER_NAME}
echo "▶| [STEP 2/5 STOP >>>>>>]"

# 💥 STEP 3: 五折多模型联合滑窗概率推理融合
echo "▶| [STEP 3/5 START >>>>>>] 汇总完成！正在调用 5 个 Fold 权重对全集执行标准推理融合..."
TOTAL_SAMPLES=$(ls -1 ${RAW_IMAGES}/*.nii.gz 2>/dev/null | wc -l || echo "0")
echo "📊 系统盘点：训练全集 [ ${DATASET_NAME} ] 共有 ${TOTAL_SAMPLES} 个待预测文件。"

# 建立输出目标目录确保安全
mkdir -p "${ENSEMBLE_DIR}"

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} nnUNetv2_predict \
  -d ${DATASET_NAME} \
  -i ${RAW_IMAGES} \
  -o ${ENSEMBLE_DIR} \
  -f 0 1 2 3 4 \
  -tr ${TRAINER_NAME} \
  -c ${CONFIG_NAME} \
  -p nnUNetPlans 
echo "▶| [STEP 3/5 STOP >>>>>>]"
echo "-----------------------------------------------------------------"

# 💥 STEP 4: 完美应用由五折自动定制生成的极致后处理 pkl 字典
echo "▶| [STEP 4/5 START >>>>>>] 推理完成！正在从五折汇总现场提取后处理决策优化图像..."
mkdir -p "${POSTPROCESSED_DIR}"
nnUNetv2_apply_postprocessing \
  -i ${ENSEMBLE_DIR} \
  -o ${POSTPROCESSED_DIR} \
  -pp_pkl_file ${MODEL_DIR}/crossval_results_folds_0_1_2_3_4/postprocessing.pkl \
  -np ${NUM_THREADS} \
  -plans_json ${MODEL_DIR}/crossval_results_folds_0_1_2_3_4/plans.json
echo "▶| [STEP 4/5 STOP >>>>>>]"
echo "-----------------------------------------------------------------"

# 💥 STEP 5: 核心修复：调用符合新版规范的 -pfile 和 -djfile 参数进行权威评估算分
echo "▶| [STEP 5/5 START >>>>>>] 后处理完成！正在调用医学图像评估器核对全集最终交叉验证学术指标..."
nnUNetv2_evaluate_folder \
  ${RAW_LABELS} \
  ${POSTPROCESSED_DIR} \
  -djfile ${MODEL_DIR}/dataset.json \
  -pfile ${MODEL_DIR}/plans.json
echo "▶| [STEP 5/5 STOP >>>>>>]"

echo "====================================================================================="
echo "🏆 🎉 五折交叉验证大满贯数据集通用看板流水线已全线完美通关！"
echo "👉 最终权威交叉验证学术战报： ${POSTPROCESSED_DIR}/summary.json"
echo "====================================================================================="