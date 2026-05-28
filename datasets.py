import os
import numpy as np
from torch.utils.data import Dataset
import torch
from typing import List

from utils import get_contents_in_dir


class TrafficScopeDataset(Dataset):
    def __init__(self, data_dir, agg_scales: List[int], indices=None):
        # load data and gen labels
        # 支持两种目录结构:
        # 1. data_dir/label/xxx.npy (新结构)
        # 2. data_dir/xxx.npy (旧结构)

        # 检查是否是新的目录结构 (data_dir/label/xxx.npy)
        label_dirs = get_contents_in_dir(data_dir, [], [])
        label_dirs = [d for d in label_dirs if os.path.isdir(d)]

        temporal_data_files = []
        temporal_mask_files = []
        contextual_data_files = []

        if label_dirs:
            # 新目录结构: data_dir/label/xxx.npy
            for label_dir in sorted(label_dirs):
                label_name = os.path.basename(label_dir)
                t_files = get_contents_in_dir(label_dir, ['.'], ['_temporal.npy'])
                m_files = get_contents_in_dir(label_dir, ['.'], ['_mask.npy'])
                c_files = get_contents_in_dir(label_dir, ['.'], ['_contextual.npy'])
                for f in t_files:
                    temporal_data_files.append((f, label_name))
                for f in m_files:
                    temporal_mask_files.append((f, label_name))
                for f in c_files:
                    contextual_data_files.append((f, label_name))
        else:
            # 旧目录结构: data_dir/xxx.npy
            t_files = get_contents_in_dir(data_dir, ['.'], ['_temporal.npy'])
            m_files = get_contents_in_dir(data_dir, ['.'], ['_mask.npy'])
            c_files = get_contents_in_dir(data_dir, ['.'], ['_contextual.npy'])
            for f in t_files:
                temporal_data_files.append((f, os.path.basename(data_dir)))
            for f in m_files:
                temporal_mask_files.append((f, os.path.basename(data_dir)))
            for f in c_files:
                contextual_data_files.append((f, os.path.basename(data_dir)))

        # 创建标签映射
        unique_labels = sorted(list(set([item[1] for item in temporal_data_files])))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        tmp_temporal_data_list = []
        tmp_temporal_mask_data_list = []
        tmp_contextual_data_list = []
        data_len = 0
        for idx in range(len(temporal_data_files)):
            file_path, label = temporal_data_files[idx]
            tmp_temporal_data = np.load(file_path)
            tmp_temporal_mask_data = np.load(temporal_mask_files[idx][0])
            tmp_contextual_data = np.load(contextual_data_files[idx][0])
            tmp_temporal_data_list.append(tmp_temporal_data)
            tmp_temporal_mask_data_list.append(tmp_temporal_mask_data)
            tmp_contextual_data_list.append(tmp_contextual_data)
            data_len += tmp_temporal_data.shape[0]
            print(f'load {file_path} (label: {label}) successfully, len: {tmp_temporal_data.shape[0]}')

        self.temporal_data = np.zeros((data_len,
                                       tmp_temporal_data_list[0].shape[1], tmp_temporal_data_list[0].shape[2]))
        self.temporal_mask_data = np.zeros((data_len,
                                            tmp_temporal_mask_data_list[0].shape[1],
                                            tmp_temporal_mask_data_list[0].shape[2]))
        self.contextual_data = np.zeros((data_len,
                                         tmp_contextual_data_list[0].shape[1],
                                         tmp_contextual_data_list[0].shape[2], tmp_contextual_data_list[0].shape[3]))
        self.labels = np.ones(data_len)
        idx = 0
        for i in range(len(tmp_temporal_data_list)):
            tmp_temporal_data = tmp_temporal_data_list[i]
            tmp_temporal_mask_data = tmp_temporal_mask_data_list[i]
            tmp_contextual_data = tmp_contextual_data_list[i]
            label_name = temporal_data_files[i][1]
            self.temporal_data[idx:idx+tmp_temporal_data.shape[0], :, :] = tmp_temporal_data[:, :, :]
            self.temporal_mask_data[idx:idx+tmp_temporal_mask_data.shape[0], :, :] = tmp_temporal_mask_data[:, :, :]
            self.contextual_data[idx:idx+tmp_contextual_data.shape[0], :, :, :] = tmp_contextual_data[:, :, :, :]
            self.labels[idx:idx+tmp_temporal_data.shape[0]] = label_to_idx[label_name]
            idx += tmp_temporal_data.shape[0]

        print(f'total len: {data_len}')
        print(f'class list: {unique_labels}')

        # gen valid len
        self.temporal_valid_len = self.temporal_mask_data.shape[1] - \
                                  (self.temporal_mask_data.sum(axis=2) == self.temporal_mask_data.shape[2]).sum(axis=1)

        # unpack contextual feature N x agg_scale_num x freqs x t --> N x freqs x (agg_scale_num x t)
        concatenate_data = []
        for agg_scale in agg_scales:
            concatenate_data.append(self.contextual_data[:, agg_scale, :, :])
        self.contextual_data_unpack = np.concatenate(concatenate_data, axis=2)
        # let the last dim be features N x freqs x (agg_scale_num x t) --> N x (agg_scale_num x t) x freqs
        self.contextual_data_unpack = self.contextual_data_unpack.transpose((0, 2, 1))

        # gen contextual segments N x (agg_scale_num x t)
        segment_len = self.contextual_data.shape[3]
        self.contextual_segments = np.zeros((self.contextual_data.shape[0],
                                             len(agg_scales)*segment_len))
        for agg_scale_idx, _ in enumerate(agg_scales):
            self.contextual_segments[:, segment_len*agg_scale_idx:segment_len*(agg_scale_idx+1)] = agg_scale_idx

        if indices is not None:
            self.temporal_data = self.temporal_data[indices]
            self.temporal_mask_data = self.temporal_mask_data[indices]
            self.contextual_data_unpack = self.contextual_data_unpack[indices]
            self.contextual_segments = self.contextual_segments[indices]
            self.labels = self.labels[indices]
            print(f'total len after indices: {self.temporal_data.shape[0]}')
        print('load dataset successfully')

    def __getitem__(self, idx):
        return torch.tensor(self.temporal_data[idx], dtype=torch.float), \
               torch.tensor(self.temporal_valid_len[idx], dtype=torch.float), \
               torch.tensor(self.contextual_data_unpack[idx], dtype=torch.float), \
               torch.tensor(self.contextual_segments[idx], dtype=torch.long), \
               torch.tensor(int(self.labels[idx]), dtype=torch.long)

    def __len__(self):
        return self.temporal_data.shape[0]


if __name__ == '__main__':
    dataset = TrafficScopeDataset('/XXX', [0, 1, 2], [0, 1, 2, 3, 4, 5])
    print(len(dataset))
