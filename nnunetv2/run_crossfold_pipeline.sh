#!/bin/bash
# ------------------------------------------------------------------------
# nnU-Net v2 【五折交叉训练 + 自动寻优 + 推理融合 + 后处理 + 权威评估】数据集通用脚本
# ------------------------------------------------------------------------
set -e # 🛡️ 数值安全防线：任何一步报错，立刻强行熔断退出

# 🌟================== 1. 数据集与通用核心变量配置区 ==================🌟
# 当需要切换到其他数据集时，只需要修改下面这两行核心变量即可！
DATASET_NAME="Dataset515_ICH2023"
DATASET_ID="515"

# 基础架构配置（跑你的 SegMamba 时，将 TRAINER_NAME 改为 nnUNetTrainerSegMamba）
CONFIG_NAME="3d_fullres"
TRAINER_NAME="nnUNetTrainer"  
# ====================================================================🌟

# 🌟================== 2. 显卡与运算资源自定义配置区 ==================🌟
# 单卡设为 "5"；若后续 SegMamba 骨干热身完毕切到多卡，直接改写为 "1,2" 即可
GPU_DEVICES="5"
NUM_THREADS=8  # 后处理与数据增强使用的 CPU 线程数
# ====================================================================🌟

# 🚀 3. 基于变量全自动派生绝对路径（完美适配任何数据集）
MODEL_DIR="/home/wjx/CodeData/code/nnUNet/nnUNet_results/${DATASET_NAME}/${TRAINER_NAME}__nnUNetPlans__${CONFIG_NAME}"
RAW_IMAGES="/home/wjx/CodeData/code/nnUNet/nnUNet_raw/${DATASET_NAME}/imagesTr" 
RAW_LABELS="/home/wjx/CodeData/code/nnUNet/nnUNet_raw/${DATASET_NAME}/labelsTr" 

# 创建五折融合输出目标文件夹
ENSEMBLE_DIR="${MODEL_DIR}/Ensemble"
POSTPROCESSED_DIR="${MODEL_DIR}/Ensemble_PostProcessing"

# 🌟=================================================================🌟
# 📊 【全功能参数大满贯全景高亮看板区 —— 五折交叉验证专用版】
# 🌟=================================================================🌟
echo "====================================================================================="
echo "🪐 正在拉起 nnU-Net v2 Cross-Fold 学术大满贯数据集通用流水线..."
echo "====================================================================================="
echo "📋 [五折核心配置环境盘点]:"
echo "   ├─ 📦 DATASET_NAME     : ${DATASET_NAME} (ID: ${DATASET_ID})"
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
    
    # 判断是否为多卡训练，自动适配 nnU-Net DDP 传参机制
    if [[ $GPU_DEVICES == *","* ]]; then
        NUM_GPUS=$(echo $GPU_DEVICES | tr -cd ',' | wc -c)
        NUM_GPUS=$((NUM_GPUS + 1))
        CUDA_VISIBLE_DEVICES=${GPU_DEVICES} nnUNetv2_train ${DATASET_NAME} ${CONFIG_NAME} ${fold} -tr ${TRAINER_NAME} -num_gpus ${NUM_GPUS}
    else
        CUDA_VISIBLE_DEVICES=${GPU_DEVICES} nnUNetv2_train ${DATASET_NAME} ${CONFIG_NAME} ${fold} -tr ${TRAINER_NAME}
    fi
done

# 💥 STEP 2: 强行打通五折验证前向汇总并搜寻最佳后处理 (生成 postprocessing.pkl)
echo "▶️ [STEP 2/5] 五折训练圆满结束！正在执行全折前向汇总与后处理决策寻优..."
nnUNetv2_find_best_configuration ${DATASET_ID} -c ${CONFIG_NAME} -tr ${TRAINER_NAME}

# 💥 STEP 3: 五折多模型滑窗概率联合推理（注入样本计数看板）
echo "▶️ [STEP 3/5] 汇总完成！正在调用 5 个 Fold 权重对全集执行标准推理融合..."

# 动态计算当前指定数据集训练集下的样本总数
TOTAL_SAMPLES=$(ls -1 ${RAW_IMAGES}/*.nii.gz 2>/dev/null | wc -l || echo "0")
echo "📊 系统盘点：训练全集 [ ${DATASET_NAME} ] 共有 ${TOTAL_SAMPLES} 个待预测文件。"
echo "-----------------------------------------------------------------"

# 执行五折集成滑窗推理，同时利用 awk 实时输出精美进度面板
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} nnUNetv2_predict \
  -d ${DATASET_NAME} \
  -i ${RAW_IMAGES} \
  -o ${ENSEMBLE_DIR} \
  -f 0 1 2 3 4 \
  -tr ${TRAINER_NAME} \
  -c ${CONFIG_NAME} \
  -p nnUNetPlans | awk -v total="${TOTAL_SAMPLES}" '
    BEGIN { count = 0; }
    /Predicting/ || /\.nii\.gz/ { 
        if ($0 ~ /Predicting/) {
            count++;
            print "\n🔮 [大融合预测进度: " count "/" total " ] 正在融合处理目标样本 ──> " $0;
            fflush();
        }
    }
    !/Predicting/ { print $0; fflush(); }
'

echo "-----------------------------------------------------------------"
# 💥 STEP 4: 应用全自动定制出的极致后处理
echo "▶️ [STEP 4/5] 推理完成！正在注入后处理决策进行图像深度优化..."
nnUNetv2_apply_postprocessing \
  -i ${ENSEMBLE_DIR} \
  -o ${POSTPROCESSED_DIR} \
  -pp_pkl_file ${MODEL_DIR}/crossval_results_folds_0_1_2_3_4/postprocessing.pkl \
  -np ${NUM_THREADS} \
  -plans_json ${MODEL_DIR}/crossval_results_folds_0_1_2_3_4/plans.json

# 💥 STEP 5: 调用评估器收割最终战报
echo "▶️ [STEP 5/5] 后处理完成！正在调用医学图像评估器核对全集最终交叉验证学术指标..."
nnUNetv2_evaluate_folder \
  ${RAW_LABELS} \
  ${POSTPROCESSED_DIR} \
  -dj ${MODEL_DIR}/dataset.json

echo "====================================================================================="
echo "🏆 🎉 五折交叉验证大满贯数据集通用看板流水线已全线通过！"
echo "📊 请直接前往查看你的终极学术战报 JSON 成果文件："
echo "👉 ${POSTPROCESSED_DIR}/summary.json"
echo "====================================================================================="