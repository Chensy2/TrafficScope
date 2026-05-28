#!/usr/bin/env python3
"""
数据处理脚本 - 针对已经是会话级别的PCAP文件
数据结构: $root_dir/$label/*.pcap (每个pcap是一个会话)
"""

import argparse
import binascii
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pywt

# ==================== 配置参数 ====================
ROOT_DIR = "/media/store/csy_data/ITC-Net-Blend-60/scenario_A_filter_pcap"
OUTPUT_DIR = "/media/store/csy_data/ITC-Net-Blend-60/TrafficScope_data/processed_data"
SESSION_LEN = 64
PACKET_LEN = 64
PACKET_OFFSET = 14
WAVE_NAME = 'cgau8'
AGG_POINTS_NUM = 128


def parse_session_pcap_to_matrix(session_pcap_path, session_len=SESSION_LEN,
                                  packet_len=PACKET_LEN, packet_offset=PACKET_OFFSET):
    """解析单个会话PCAP文件为矩阵"""
    try:
        with open(session_pcap_path, 'rb') as f:
            content = f.read()
    except Exception as e:
        print(f'Error reading {session_pcap_path}: {e}')
        return None, None

    hexc = binascii.hexlify(content)

    # 检查字节序
    if hexc[:8] == b'd4c3b2a1':
        little_endian = True
    else:
        little_endian = False

    # 移除全局包头 24字节
    hexc = hexc[48:]

    # 解析数据包
    packets_dec = []
    while len(hexc) > 0 and len(packets_dec) < session_len:
        if len(hexc) < 32:
            break
        frame_len = hexc[16:24]
        if little_endian:
            frame_len = binascii.hexlify(binascii.unhexlify(frame_len)[::-1])
        frame_len = int(frame_len, 16)

        hexc = hexc[32:]  # 移除包头 16字节
        frame_hex = hexc[packet_offset * 2:min(packet_len * 2, frame_len * 2)]
        frame_dec = [int(frame_hex[i:i + 2], 16) for i in range(0, len(frame_hex), 2)]
        if frame_dec:
            packets_dec.append(frame_dec)

        hexc = hexc[frame_len * 2:]

    if len(packets_dec) < 3:
        return None, None

    # 填充并构建会话矩阵
    packets_dec_matrix = pd.DataFrame(packets_dec).fillna(-1).values.astype(np.int8)
    session_matrix = np.ones((session_len, packet_len), dtype=np.int8) * -1
    row_idx = min(packets_dec_matrix.shape[0], session_len)
    col_idx = min(packets_dec_matrix.shape[1], packet_len)
    session_matrix[:row_idx, :col_idx] = packets_dec_matrix[:row_idx, :col_idx]

    return session_matrix, (session_matrix == -1).astype(np.uint8)


def wavelet_transform(seq, wave_name, agg_points_num):
    """小波变换生成频谱图"""
    scales = np.arange(1, agg_points_num + 1)
    fc = pywt.central_frequency(wave_name)
    scales = 2 * fc * agg_points_num / scales
    cwtmatr, freqs = pywt.cwt(seq, scales, wave_name)
    spectrogram = np.log2((abs(cwtmatr)) ** 2 + 1)
    spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) + 1)
    return spectrogram


def gen_contextual_from_packets(session_matrix, wave_name, agg_points_num=AGG_POINTS_NUM):
    """从时域数据生成小波域特征"""
    # 计算每行（每个包）的非填充值总和作为"长度序列"
    packet_lengths = []
    for row in session_matrix:
        valid_values = row[row != -1]
        if len(valid_values) > 0:
            packet_lengths.append(np.sum(valid_values))
        else:
            packet_lengths.append(0)

    seq = np.array(packet_lengths, dtype=float)

    # 生成三个时间尺度的频谱图
    ms_spectrogram = wavelet_transform(seq, wave_name, agg_points_num)
    s_spectrogram = wavelet_transform(seq, wave_name, agg_points_num)
    min_spectrogram = wavelet_transform(seq, wave_name, agg_points_num)

    # 堆叠成 3 x freqs x time
    contextual = np.stack([ms_spectrogram, s_spectrogram, min_spectrogram], axis=0)
    return contextual


def process_single_label(label_dir, label_name, output_dir):
    """处理单个类别的所有PCAP文件"""
    print(f"\n{'='*50}")
    print(f"处理类别: {label_name}")
    print(f"{'='*50}")

    # 找出所有pcap文件
    pcap_files = list(Path(label_dir).glob("*.pcap")) + list(Path(label_dir).glob("*.pcapng"))

    if not pcap_files:
        print(f"警告: {label_dir} 中没有找到pcap文件，跳过")
        return 0

    print(f"找到 {len(pcap_files)} 个会话文件")

    # 创建输出目录
    label_output_dir = Path(output_dir) / label_name
    label_output_dir.mkdir(parents=True, exist_ok=True)

    # 存储数据
    temporal_data_list = []
    temporal_mask_list = []
    contextual_data_list = []
    session_files_used = []

    parse_start = time.time()

    for idx, pcap_file in enumerate(pcap_files):
        session_matrix, padding_mask = parse_session_pcap_to_matrix(str(pcap_file))

        if session_matrix is None:
            print(f"跳过 (太短): {pcap_file.name}")
            continue

        temporal_data_list.append(session_matrix)
        temporal_mask_list.append(padding_mask)

        # 生成小波域特征
        contextual = gen_contextual_from_packets(session_matrix, WAVE_NAME)
        contextual_data_list.append(contextual)

        session_files_used.append(str(pcap_file))

        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{len(pcap_files)} 个文件...")

    parse_end = time.time()

    if not temporal_data_list:
        print(f"错误: {label_name} 没有有效的会话数据")
        return 0

    # 转换为numpy数组
    temporal_data = np.array(temporal_data_list, dtype=np.int8)
    temporal_mask = np.array(temporal_mask_list, dtype=np.uint8)
    contextual_data = np.array(contextual_data_list, dtype=np.float32)

    print(f"\n有效会话数: {len(temporal_data)}")
    print(f"时域数据形状: {temporal_data.shape}")
    print(f"小波域数据形状: {contextual_data.shape}")
    print(f"处理时间: {parse_end - parse_start:.2f}秒")

    # 保存数据
    temporal_path = label_output_dir / f"{label_name}_temporal.npy"
    mask_path = label_output_dir / f"{label_name}_temporal_mask.npy"
    contextual_path = label_output_dir / f"{label_name}_{WAVE_NAME}_contextual.npy"
    session_list_path = label_output_dir / f"{label_name}_session_used.json"

    np.save(temporal_path, temporal_data)
    np.save(mask_path, temporal_mask)
    np.save(contextual_path, contextual_data)

    with open(session_list_path, 'w') as f:
        json.dump(session_files_used, f)

    print(f"已保存:")
    print(f"  - {temporal_path}")
    print(f"  - {mask_path}")
    print(f"  - {contextual_path}")
    print(f"  - {session_list_path}")

    return len(temporal_data)


def main():
    # 创建输出目录
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("="*50)
    print("开始处理数据集")
    print(f"输入目录: {ROOT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*50)

    total_sessions = 0
    label_stats = {}

    # 遍历每个类别目录
    root_path = Path(ROOT_DIR)
    for label_dir in sorted(root_path.iterdir()):
        if not label_dir.is_dir():
            continue

        label_name = label_dir.name
        count = process_single_label(label_dir, label_name, OUTPUT_DIR)

        if count > 0:
            total_sessions += count
            label_stats[label_name] = count

    # 总结
    print("\n" + "="*50)
    print("处理完成！")
    print(f"总类别数: {len(label_stats)}")
    print(f"总有效会话数: {total_sessions}")
    print("\n各类别样本数:")
    for label, count in sorted(label_stats.items()):
        print(f"  {label}: {count}")
    print("="*50)


if __name__ == '__main__':
    main()
