#!/bin/bash
# ------------------------------------------------------------------------
# nnU-Net v2 【全量训练 + 独立测试集推理 + 后处理 + 极限评估】数据集通用脚本
# ------------------------------------------------------------------------
set -e # 🛡️ 数值安全防线：任何一步报错，立刻强行熔断退出

# 🌟================== 1. 数据集与通用核心变量配置区 ==================🌟
# 当需要切换到其他数据集时，只需要修改下面这两行核心变量即可！
DATASET_NAME="Dataset515_ICH2023"
DATASET_ID="515"

# 基础架构配置（跑你的 SegMamba 时，将 TRAINER_NAME 改为 nnUNetTrainerSegMamba）
CONFIG_NAME="segmamba"
TRAINER_NAME="nnUNetTrainerSegMamba"  
# ====================================================================🌟

# 🌟================== 2. 显卡与运算资源自定义配置区 ==================🌟
# 单卡设为 "5"；若后续 SegMamba 骨干热身完毕切到多卡，直接改写为 "1,2" 即可
GPU_DEVICES="5"
NUM_THREADS=8  # 后处理与数据增强使用的 CPU 线程数
# ====================================================================🌟

# 🚀 3. 基于变量全自动派生绝对路径（完美适配任何数据集）
MODEL_DIR="/home/wjx/CodeData/code/nnUNet/nnUNet_results/${DATASET_NAME}/${TRAINER_NAME}__nnUNetPlans__${CONFIG_NAME}"
TEST_IMAGES="/home/wjx/CodeData/code/nnUNet/nnUNet_raw/${DATASET_NAME}/imagesTs" # 动态测试集原图
TEST_LABELS="/home/wjx/CodeData/code/nnUNet/nnUNet_raw/${DATASET_NAME}/labelsTs" # 动态测试集真实标签

# 创建测试输出目标文件夹
TEST_PRED_DIR="${MODEL_DIR}/Test_predictions"
TEST_PRED_PP_DIR="${MODEL_DIR}/Test_predictions_PostProcessing"

echo "================================================================="
echo "🪐 正在拉起 nnU-Net v2 All-in-Test 生产环境通用全流程流水线..."
echo "   📦 目标数据集: ${DATASET_NAME}"
echo "   📌 当前绑定显卡: CUDA_VISIBLE_DEVICES=${GPU_DEVICES}"
echo "================================================================="

# 💥 STEP 1: 全量数据长跑训练 (Fold = all)
echo "▶️ [STEP 1/4] 正在拉起全量数据训练模型。注意：此模式下无验证集评估..."
if [[ $GPU_DEVICES == *","* ]]; then
    echo "⚡ 检测到多卡配置，自动激活分布式 DDP 训练模式..."
    NUM_GPUS=$(echo $GPU_DEVICES | tr -cd ',' | wc -c)
    NUM_GPUS=$((NUM_GPUS + 1))
    CUDA_VISIBLE_DEVICES=${GPU_DEVICES} nnUNetv2_train ${DATASET_NAME} ${CONFIG_NAME} all -tr ${TRAINER_NAME} -num_gpus ${NUM_GPUS}
else
    echo "⚡ 激活单卡训练模式..."
    CUDA_VISIBLE_DEVICES=${GPU_DEVICES} nnUNetv2_train ${DATASET_NAME} ${CONFIG_NAME} all -tr ${TRAINER_NAME}
fi

# 💥 STEP 2: 独立测试集全量滑窗推理（注入样本计数看板）
echo "▶️ [STEP 2/4] 训练圆满结束！正在对独立测试集进行推理预测..."

# 动态计算当前指定数据集下的测试样本总数
TOTAL_SAMPLES=$(ls -1 ${TEST_IMAGES}/*.nii.gz 2>/dev/null | wc -l || echo "0")
echo "📊 系统盘点：独立测试集 [ ${DATASET_NAME} ] 共有 ${TOTAL_SAMPLES} 个待预测文件。"
echo "-----------------------------------------------------------------"

# 执行多线程推理并通过 awk 实时过滤输出流，优雅高亮显示 [当前/总数] 进度
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} nnUNetv2_predict \
  -d ${DATASET_NAME} \
  -i ${TEST_IMAGES} \
  -o ${TEST_PRED_DIR} \
  -f all \
  -tr ${TRAINER_NAME} \
  -c ${CONFIG_NAME} \
  -p nnUNetPlans | awk -v total="${TOTAL_SAMPLES}" '
    BEGIN { count = 0; }
    /Predicting/ || /\.nii\.gz/ { 
        if ($0 ~ /Predicting/) {
            count++;
            print "\n🔮 [测试集预测进度: " count "/" total " ] 正在滑窗推理目标样本 ──> " $0;
            fflush();
        }
    }
    !/Predicting/ { print $0; fflush(); }
'

echo "-----------------------------------------------------------------"
# 💥 STEP 3: 移植最稳健的后处理配置
echo "▶️ [STEP 3/4] 推理完成！正在从 fold_all 现场提取后处理决策并深度优化测试集图像..."
nnUNetv2_apply_postprocessing \
  -i ${TEST_PRED_DIR} \
  -o ${TEST_PRED_PP_DIR} \
  -pp_pkl_file ${MODEL_DIR}/fold_all/postprocessing.pkl \
  -np ${NUM_THREADS} \
  -plans_json ${MODEL_DIR}/fold_all/plans.json

# 💥 STEP 4: 在独立测试集上进行全量学术指标大盘点
echo "▶️ [STEP 4/4] 后处理完成！正在调用医学图像评估器严格核对测试集最终 Dice 指标..."
nnUNetv2_evaluate_folder \
  ${TEST_LABELS} \
  ${TEST_PRED_PP_DIR} \
  -dj ${MODEL_DIR}/dataset.json

echo "================================================================="
echo "🏆 🎉 恭喜！【训练->测试->后处理->评估】变量参数化通用流水线完美通关！"
echo "📊 请前往查看你在独立测试集（Test Set）上的最终王牌战报文件："
echo "👉 ${TEST_PRED_PP_DIR}/summary.json"
echo "================================================================="
