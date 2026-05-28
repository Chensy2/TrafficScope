#!/bin/bash
#SBATCH --job-name=train-trafficscope
#SBATCH --time=120:00:00              # 时间限制
#SBATCH --partition=gpu               # 使用GPU分区
#SBATCH --exclude=GPU41
#SBATCH --gres=gpu:1                  # 请求1个GPU
#SBATCH --mem=64G                     # 内存
#SBATCH --nodes=1                     # 节点数
#SBATCH --ntasks=1                    # 任务数
#SBATCH --cpus-per-task=4             # 每任务CPU核数
#SBATCH --output=./outputs/trafficscope/train_%j.out
#SBATCH --error=./outputs/trafficscope/train_%j.err

# 打印信息
echo "Submitting job from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# 初始化conda
eval "$(conda shell.bash hook)"

# 激活环境
conda activate myflowenv

# ==================== 超参数配置 ====================
# 数据路径
DATA_DIR="/media/store/csy_data/ITC-Net-Blend-60/TrafficScope_data/ITC_Net_A"

# 模型配置 (修改这里选择不同模型)
# 选项: temporal, contextual, fusion
MODEL_TYPE="fusion"

# 训练超参数
NUM_CLASSES=60
BATCH_SIZE=32
EPOCHS=10
LR=0.001
NUM_HEADS=8
NUM_LAYERS=2
DROPOUT=0.5

# 输出路径
MODEL_DIR="./models"
RESULT_DIR="./results"

# 创建输出目录
mkdir -p outputs/trafficscope
mkdir -p "$MODEL_DIR"
mkdir -p "$RESULT_DIR"

# ==================== 根据模型类型设置参数 ====================
case "$MODEL_TYPE" in
    temporal)
        echo "训练模型: 仅时域 (TrafficScopeTemporal)"
        USE_TEMPORAL="--use_temporal"
        USE_CONTEXTUAL=""
        MODEL_PATH="$MODEL_DIR/temporal_only.pth"
        RESULT_PATH="$RESULT_DIR/temporal_only.npy"
        ;;
    contextual)
        echo "训练模型: 仅小波域 (TrafficScopeContextual)"
        USE_TEMPORAL=""
        USE_CONTEXTUAL="--use_contextual"
        MODEL_PATH="$MODEL_DIR/contextual_only.pth"
        RESULT_PATH="$RESULT_DIR/contextual_only.npy"
        ;;
    fusion)
        echo "训练模型: 融合模型 (TrafficScope)"
        USE_TEMPORAL="--use_temporal"
        USE_CONTEXTUAL="--use_contextual"
        MODEL_PATH="$MODEL_DIR/full_fusion.pth"
        RESULT_PATH="$RESULT_DIR/full_fusion.npy"
        ;;
    *)
        echo "错误: MODEL_TYPE必须是 'temporal', 'contextual' 或 'fusion'"
        exit 1
        ;;
esac

echo "======================================"
echo "配置:"
echo "  模型类型: $MODEL_TYPE"
echo "  类别数: $NUM_CLASSES"
echo "  批大小: $BATCH_SIZE"
echo "  训练轮数: $EPOCHS"
echo "  学习率: $LR"
echo "  模型保存: $MODEL_PATH"
echo "  结果保存: $RESULT_PATH"
echo "======================================"

# ==================== 训练命令 ====================
python train_test.py \
    --data_dir="$DATA_DIR" \
    $USE_TEMPORAL \
    $USE_CONTEXTUAL \
    --is_train \
    --is_test \
    --num_classes=$NUM_CLASSES \
    --batch_size=$BATCH_SIZE \
    --epochs=$EPOCHS \
    --lr=$LR \
    --num_heads=$NUM_HEADS \
    --num_layers=$NUM_LAYERS \
    --dropout=$DROPOUT \
    --model_path="$MODEL_PATH" \
    --result_path="$RESULT_PATH"

echo "训练完成！"
