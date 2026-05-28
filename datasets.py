import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import get_contents_in_dir


class TrafficScopeDataset(Dataset):
    def __init__(self, data_dir, agg_scales: List[int], indices=None):
        def collect_records(search_dir, default_label=None):
            records = []
            t_files = get_contents_in_dir(search_dir, ['.'], ['_temporal.npy'])
            t_files = [f for f in t_files if f.endswith('_temporal.npy')]
            all_npy = get_contents_in_dir(search_dir, ['.'], ['.npy'])
            all_npy = [f for f in all_npy if f.endswith('.npy')]

            for temporal_path in sorted(t_files):
                temporal_name = os.path.basename(temporal_path)
                base_name = temporal_name[:-len('_temporal.npy')]
                mask_candidates = [
                    os.path.join(search_dir, f'{base_name}_temporal_mask.npy'),
                    os.path.join(search_dir, f'{base_name}_mask.npy')
                ]
                mask_path = next((p for p in mask_candidates if os.path.exists(p)), None)
                contextual_candidates = [
                    p for p in all_npy
                    if os.path.basename(p).startswith(f'{base_name}_') and p.endswith('_contextual.npy')
                ]

                if mask_path is None:
                    raise FileNotFoundError(f'Cannot find mask file for {temporal_path}')
                if len(contextual_candidates) != 1:
                    raise FileNotFoundError(f'Cannot uniquely find contextual file for {temporal_path}')

                records.append({
                    'temporal': temporal_path,
                    'mask': mask_path,
                    'contextual': contextual_candidates[0],
                    'label': default_label if default_label is not None else base_name
                })
            return records

        label_dirs = get_contents_in_dir(data_dir, [], [])
        label_dirs = [d for d in label_dirs if os.path.isdir(d)]

        data_records = []
        if label_dirs:
            for label_dir in sorted(label_dirs):
                data_records.extend(collect_records(label_dir, os.path.basename(label_dir)))
        else:
            data_records.extend(collect_records(data_dir))

        if not data_records:
            raise FileNotFoundError(f'Cannot find TrafficScope npy files in {data_dir}')

        unique_labels = sorted(list(set([item['label'] for item in data_records])))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        tmp_temporal_data_list = []
        tmp_temporal_mask_data_list = []
        tmp_contextual_data_list = []
        data_len = 0
        for record in data_records:
            tmp_temporal_data = np.load(record['temporal'])
            tmp_temporal_mask_data = np.load(record['mask'])
            tmp_contextual_data = np.load(record['contextual'])
            if tmp_temporal_data.shape[0] != tmp_temporal_mask_data.shape[0] or \
                    tmp_temporal_data.shape[0] != tmp_contextual_data.shape[0]:
                raise ValueError(f'sample count mismatch in {record}')
            tmp_temporal_data_list.append(tmp_temporal_data)
            tmp_temporal_mask_data_list.append(tmp_temporal_mask_data)
            tmp_contextual_data_list.append(tmp_contextual_data)
            data_len += tmp_temporal_data.shape[0]
            print(f"load {record['temporal']} (label: {record['label']}) successfully, "
                  f"len: {tmp_temporal_data.shape[0]}")

        self.temporal_data = np.zeros((data_len,
                                       tmp_temporal_data_list[0].shape[1], tmp_temporal_data_list[0].shape[2]))
        self.temporal_mask_data = np.zeros((data_len,
                                            tmp_temporal_mask_data_list[0].shape[1],
                                            tmp_temporal_mask_data_list[0].shape[2]))
        self.contextual_data = np.zeros((data_len,
                                         tmp_contextual_data_list[0].shape[1],
                                         tmp_contextual_data_list[0].shape[2],
                                         tmp_contextual_data_list[0].shape[3]))
        self.labels = np.ones(data_len)
        idx = 0
        for i in range(len(tmp_temporal_data_list)):
            tmp_temporal_data = tmp_temporal_data_list[i]
            tmp_temporal_mask_data = tmp_temporal_mask_data_list[i]
            tmp_contextual_data = tmp_contextual_data_list[i]
            label_name = data_records[i]['label']
            self.temporal_data[idx:idx+tmp_temporal_data.shape[0], :, :] = tmp_temporal_data[:, :, :]
            self.temporal_mask_data[idx:idx+tmp_temporal_mask_data.shape[0], :, :] = tmp_temporal_mask_data[:, :, :]
            self.contextual_data[idx:idx+tmp_contextual_data.shape[0], :, :, :] = tmp_contextual_data[:, :, :, :]
            self.labels[idx:idx+tmp_temporal_data.shape[0]] = label_to_idx[label_name]
            idx += tmp_temporal_data.shape[0]

        print(f'total len: {data_len}')
        print(f'class list: {unique_labels}')

        self.temporal_valid_len = self.temporal_mask_data.shape[1] - \
            (self.temporal_mask_data.sum(axis=2) == self.temporal_mask_data.shape[2]).sum(axis=1)

        concatenate_data = []
        for agg_scale in agg_scales:
            concatenate_data.append(self.contextual_data[:, agg_scale, :, :])
        self.contextual_data_unpack = np.concatenate(concatenate_data, axis=2)
        self.contextual_data_unpack = self.contextual_data_unpack.transpose((0, 2, 1))

        segment_len = self.contextual_data.shape[3]
        self.contextual_segments = np.zeros((self.contextual_data.shape[0],
                                             len(agg_scales)*segment_len))
        for agg_scale_idx, _ in enumerate(agg_scales):
            self.contextual_segments[:, segment_len*agg_scale_idx:segment_len*(agg_scale_idx+1)] = agg_scale_idx

        if indices is not None:
            self.temporal_data = self.temporal_data[indices]
            self.temporal_mask_data = self.temporal_mask_data[indices]
            self.temporal_valid_len = self.temporal_valid_len[indices]
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
