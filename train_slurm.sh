#!/bin/bash
#SBATCH --job-name=train-trafficllm-csy
#SBATCH --time=120:00:00              # estimate time limit of the job
#SBATCH --partition=big            # use the 'gpu' partition
#SBATCH --gres=shard:A800:1        # request 1 GPU, or use gpu:V100:1 to specify GPU type
#SBATCH --mem=100G                      # request 2 GB RAM
#SBATCH --nodes=1                     # total number of nodes
#SBATCH --ntasks=1                    # total number of tasks
#SBATCH --cpus-per-task 4             # number of CPU cores per task      
#SBATCH --output=./logs/ITC_Net_A/train_temporal.out
#SBATCH --error=./logs/ITC_Net_A/train_temporal.err

# 打印信息
echo "Submitting job from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# 初始化conda
eval "$(conda shell.bash hook)"

# 激活环境
conda activate trafficscope


# ==================== 超参数配置 ====================

DATASET="ITC_Net_A"
# 选项: temporal, contextual, fusion
MODEL_TYPE="temporal"
# 数据路径
DATA_DIR="dataset/${DATASET}"
# 类别数
NUM_CLASSES=58
# early stopping
EPOCHS=50
PATIENCE=4

# 训练超参数
BATCH_SIZE=32
LR=0.001
NUM_HEADS=8
NUM_LAYERS=4
DROPOUT=0.5

# 数据维度
TEMPORAL_SEQ_LEN=64
PACKET_LEN=64
FREQS_SIZE=128
AGG_POINTS_NUM=128
AGG_SCALE_NUM=3

# 输出路径
MODEL_DIR="./models/${DATASET}/${MODEL_TYPE}"
RESULT_DIR="./results/${DATASET}/${MODEL_TYPE}"

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
    --patience=$PATIENCE \
    --lr=$LR \
    --num_heads=$NUM_HEADS \
    --num_layers=$NUM_LAYERS \
    --dropout=$DROPOUT \
    --model_path="$MODEL_PATH" \
    --result_path="$RESULT_PATH" \
    --temporal_seq_len=$TEMPORAL_SEQ_LEN \
    --packet_len=$PACKET_LEN \
    --agg_scale_num=$AGG_SCALE_NUM \
    --agg_points_num=$AGG_POINTS_NUM \
    --freqs_size=$FREQS_SIZE

echo "训练完成！"
