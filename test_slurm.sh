#!/bin/bash
#SBATCH --job-name=test-trafficllm-csy
#SBATCH --time=2:00:00              
#SBATCH --partition=big
#SBATCH --gres=shard:A800:1
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 4
#SBATCH --output=./logs/ITC_Net_A/test_${MODEL_TYPE}.out
#SBATCH --error=./logs/ITC_Net_A/test_${MODEL_TYPE}.err

# 打印信息
echo "Submitting job from directory: ${SLURM_SUBMIT_DIR}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# 初始化conda
eval "$(conda shell.bash hook)"

# 激活环境
conda activate trafficscope

# ==================== 超参数配置 ====================

DATASET="ITC_Net_A"
# 选项: temporal, contextual, fusion
MODEL_TYPE="temporal"  # 修改这里选择要测试的模型
# 数据路径
DATA_DIR="dataset/${DATASET}"
# 类别数
NUM_CLASSES=58

# 测试超参数
BATCH_SIZE=32

# 输入路径（已训练好的模型）
MODEL_PATH="./models/${DATASET}/${MODEL_TYPE}/${MODEL_TYPE}_only.pth"
# 输出路径
RESULT_DIR="./results/${DATASET}/${MODEL_TYPE}"
RESULT_PATH="$RESULT_DIR/${MODEL_TYPE}_only.npy"

# 创建输出目录
mkdir -p "$RESULT_DIR"

# ==================== 根据模型类型设置参数 ====================
case "$MODEL_TYPE" in
    temporal)
        echo "测试模型: 仅时域 (TrafficScopeTemporal)"
        USE_TEMPORAL="--use_temporal"
        USE_CONTEXTUAL=""
        ;;
    contextual)
        echo "测试模型: 仅小波域 (TrafficScopeContextual)"
        USE_TEMPORAL=""
        USE_CONTEXTUAL="--use_contextual"
        ;;
    fusion)
        echo "测试模型: 融合模型 (TrafficScope)"
        USE_TEMPORAL="--use_temporal"
        USE_CONTEXTUAL="--use_contextual"
        ;;
    *)
        echo "错误: MODEL_TYPE必须是 'temporal', 'contextual' 或 'fusion'"
        exit 1
        ;;
esac

echo "======================================"
echo "测试配置:"
echo "  模型类型: $MODEL_TYPE"
echo "  类别数: $NUM_CLASSES"
echo "  批大小: $BATCH_SIZE"
echo "  模型路径: $MODEL_PATH"
echo "  结果保存: $RESULT_PATH"
echo "======================================"

# ==================== 测试命令 ====================
python train_test.py \
    --data_dir="$DATA_DIR" \
    $USE_TEMPORAL \
    $USE_CONTEXTUAL \
    --is_test \
    --num_classes=$NUM_CLASSES \
    --batch_size=$BATCH_SIZE \
    --model_path="$MODEL_PATH" \
    --result_path="$RESULT_PATH"

echo "测试完成！"