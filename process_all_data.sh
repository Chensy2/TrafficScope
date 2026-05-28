#!/bin/bash
# 数据处理脚本
# 数据结构: $root_dir/$label/*.pcap
# 输出结构: $output_dir/$label/*.npy

set -e  # 遇到错误退出

# ==================== 配置参数 ====================
ROOT_DIR="/media/store/csy_data/ITC-Net-Blend-60/scenario_A_filter_pcap"
OUTPUT_DIR="/media/store/csy_data/ITC-Net-Blend-60/processed_data"
SESSIONS_DIR="/media/store/csy_data/ITC-Net-Blend-60/sessions"
WAVE_NAME="cgau8"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$SESSIONS_DIR"

echo "======================================"
echo "开始处理数据集"
echo "输入目录: $ROOT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "======================================"

# ==================== 遍历每个类别 ====================
for LABEL_DIR in "$ROOT_DIR"/*; do
    if [ ! -d "$LABEL_DIR" ]; then
        continue
    fi

    LABEL=$(basename "$LABEL_DIR")
    echo ""
    echo "======================================"
    echo "处理类别: $LABEL"
    echo "======================================"

    # 检查该目录下是否有pcap文件
    PCAP_COUNT=$(find "$LABEL_DIR" -maxdepth 1 -name "*.pcap" -o -name "*.pcapng" | wc -l)
    if [ "$PCAP_COUNT" -eq 0 ]; then
        echo "警告: $LABEL_DIR 中没有找到pcap文件，跳过"
        continue
    fi

    # 创建该类别的输出目录
    LABEL_OUTPUT_DIR="$OUTPUT_DIR/$LABEL"
    LABEL_SESSIONS_DIR="$SESSIONS_DIR/${LABEL}_sessions"
    mkdir -p "$LABEL_OUTPUT_DIR"
    mkdir -p "$LABEL_SESSIONS_DIR"

    # 生成时域特征
    echo "[$LABEL] 步骤1/2: 生成时域特征..."
    python3 dataset_gen.py \
        --pcaps_path="$LABEL_DIR" \
        --class_name="$LABEL" \
        --sessions_dir="$LABEL_SESSIONS_DIR" \
        --data_path="$LABEL_OUTPUT_DIR/$LABEL.npy" \
        --wave_name="$WAVE_NAME"

    echo "[$LABEL] 步骤2/2: 生成小波域特征..."
    # 生成小波域特征（需要使用刚刚生成的session_used.json）
    python3 dataset_gen.py \
        --contextual \
        --pcaps_path="$LABEL_OUTPUT_DIR/${LABEL}.pcap" \
        --session_pcaps_used="$LABEL_OUTPUT_DIR/${LABEL}_temporal_session_used.json" \
        --wave_name="$WAVE_NAME" \
        --data_path="$LABEL_OUTPUT_DIR/${LABEL}.npy"

    echo "[$LABEL] 完成！"
    echo ""
done

echo "======================================"
echo "所有数据处理完成！"
echo "输出目录: $OUTPUT_DIR"
echo "======================================"

# 验证生成的文件
echo ""
echo "生成的文件列表:"
find "$OUTPUT_DIR" -name "*.npy" -o -name "*_session_used.json" | sort

echo ""
echo "各类别样本数统计:"
for LABEL_DIR in "$OUTPUT_DIR"/*; do
    if [ -d "$LABEL_DIR" ]; then
        LABEL=$(basename "$LABEL_DIR")
        TEMPORAL_FILE="$LABEL_DIR/${LABEL}_temporal.npy"
        if [ -f "$TEMPORAL_FILE" ]; then
            SAMPLES=$(python3 -c "import numpy as np; print(np.load('$TEMPORAL_FILE').shape[0])" 2>/dev/null || echo "?")
            echo "  $LABEL: $SAMPLES 个会话"
        fi
    fi
done
