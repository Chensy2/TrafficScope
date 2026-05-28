import torch
from torch import nn
import math
import torch.nn.functional as F

from metaconst import TRAFFIC_SCOPE, TRAFFIC_SCOPE_TEMPORAL, TRAFFIC_SCOPE_CONTEXTUAL


# ==================== Transformer Components ====================

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(
            10000,
            torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class PositionWiseFFN(nn.Module):
    """位置前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """残差连接和层归一化"""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = self.transpose_qkv(self.W_q(queries), self.num_heads)
        keys = self.transpose_qkv(self.W_k(keys), self.num_heads)
        values = self.transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，用第一项复制有效长度，然后轴2复制num_heads次
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)
        output_concat = self.transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

    def transpose_qkv(self, X, num_heads):
        """转置为了并行计算"""
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X, num_heads):
        """逆转transpose_qkv函数"""
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)


class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩码元素来执行softmax操作"""
    # X: 3D tensor [batch_size * num_heads, seq_len, seq_len]
    # valid_lens: 1D tensor [batch_size * num_heads]
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        seq_len = shape[2]

        # 创建掩码: [batch_size * num_heads, 1, seq_len]
        mask = torch.arange(seq_len, device=X.device)[None, :] >= valid_lens[:, None]
        mask = mask[:, None, :]  # [batch_size * num_heads, 1, seq_len]

        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换
        X = X.masked_fill(mask, -1e6)
        return nn.functional.softmax(X, dim=-1)


class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(key_size, query_size, value_size,
                                             num_hiddens, num_heads, dropout,
                                             use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class Encoder(nn.Module):
    """Transformer编码器基类"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# ==================== Model Components ====================

class TemporalEncoder(Encoder):
    def __init__(self, packet_len, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TemporalEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Linear(packet_len, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, use_bias))
        self.attention_weights = [None] * len(self.blks)
        self.temporal_features = None
        self.relu = nn.ReLU()

    def forward(self, X, valid_lens, *args):
        X = self.pos_encoding(self.relu(self.embedding(X)) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        self.temporal_features = X
        return X


class ContextualEncoder(Encoder):
    def __init__(self, agg_scale_num, freqs_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(ContextualEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Linear(freqs_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.segment_encoding = nn.Embedding(agg_scale_num, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, use_bias))
        self.attention_weights = [None] * len(self.blks)
        self.contextual_features = None
        self.relu = nn.ReLU()

    def forward(self, X, contextual_segments, *args):
        X = self.pos_encoding(self.relu(self.embedding(X)) * math.sqrt(self.num_hiddens)) + \
            self.segment_encoding(contextual_segments)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, torch.ones(X.size(0), device=X.device) * X.size(1))
            self.attention_weights[i] = blk.attention.attention.attention_weights
        self.contextual_features = X
        return X


class FusionEncoder(nn.Module):
    def __init__(self, temporal_dim, contextual_dim, num_hiddens, num_heads,
                 norm_shape, ffn_num_input, ffn_num_hiddens, dropout):
        super(FusionEncoder, self).__init__()

        assert num_hiddens % num_heads == 0, 'num_hiddens should be divided by num_heads'

        self.num_heads = num_heads
        self.num_hiddens = num_hiddens
        self.depth = self.num_hiddens // self.num_heads

        self.WQ = nn.Linear(temporal_dim, num_hiddens)
        self.WK = nn.Linear(contextual_dim, num_hiddens)
        self.WV = nn.Linear(contextual_dim, num_hiddens)
        self.dropout = nn.Dropout(dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

        self.attention_weights = None
        self.fusion_features = None

    def forward(self, temporal_feature, contextual_feature):
        batch_size = temporal_feature.shape[0]

        q = self.WQ(temporal_feature)
        k = self.WK(contextual_feature)
        v = self.WV(contextual_feature)

        Q = q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        K = k.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        V = v.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        attention_scores = torch.einsum('bnid,bnjd->bnij', Q, K) / math.sqrt(self.depth)
        attention_weights = F.softmax(attention_scores, dim=-1)
        out = torch.einsum('bnij,bnjd->bnid', self.dropout(attention_weights), V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_hiddens)
        out = self.addnorm1(q, out)
        out = self.addnorm2(out, self.ffn(out))

        self.attention_weights = attention_weights
        self.fusion_features = out
        return out


class TrafficScope(nn.Module):
    def __init__(self, temporal_seq_len, packet_len,
                 freqs_size, agg_scale_num, agg_points_num,
                 num_heads, num_layers, num_classes, dropout):
        super(TrafficScope, self).__init__()
        self.model_name = TRAFFIC_SCOPE
        self.temporal_encoder = TemporalEncoder(packet_len, packet_len, packet_len, packet_len,
                                                packet_len, (temporal_seq_len, packet_len),
                                                packet_len, packet_len * 2, num_heads, num_layers, dropout)
        self.contextual_encoder = ContextualEncoder(agg_scale_num, freqs_size, freqs_size, freqs_size, freqs_size,
                                                    freqs_size, (agg_scale_num * agg_points_num, freqs_size),
                                                    freqs_size, freqs_size * 2,
                                                    num_heads, num_layers, dropout)
        self.fusion_encoder = FusionEncoder(packet_len, freqs_size, packet_len, num_heads,
                                            (temporal_seq_len, packet_len),
                                            packet_len, packet_len * 2, dropout)
        self.fc = nn.Linear(temporal_seq_len * packet_len, num_classes)

    def forward(self, temporal_data, temporal_valid_len, contextual_data, contextual_segments):
        temporal_feature = self.temporal_encoder(temporal_data, temporal_valid_len)
        contextual_feature = self.contextual_encoder(contextual_data, contextual_segments)
        out = self.fusion_encoder(temporal_feature, contextual_feature)
        out = F.softmax(self.fc(torch.flatten(out, start_dim=1)), dim=-1)
        return out

    def get_temporal_attention_weights(self):
        return self.temporal_encoder.attention_weights

    def get_temporal_features(self):
        return self.temporal_encoder.temporal_features

    def get_contextual_attention_weights(self):
        return self.contextual_encoder.attention_weights

    def get_contextual_features(self):
        return self.contextual_encoder.contextual_features

    def get_fusion_attention_weights(self):
        return self.fusion_encoder.attention_weights

    def get_fusion_features(self):
        return self.fusion_encoder.fusion_features


class TrafficScopeTemporal(nn.Module):
    def __init__(self, temporal_seq_len, packet_len,
                 num_heads, num_layers, num_classes, dropout):
        super(TrafficScopeTemporal, self).__init__()
        self.model_name = TRAFFIC_SCOPE_TEMPORAL
        self.temporal_encoder = TemporalEncoder(packet_len, packet_len, packet_len, packet_len,
                                                packet_len, (temporal_seq_len, packet_len),
                                                packet_len, packet_len * 2, num_heads, num_layers, dropout)
        self.fc = nn.Linear(temporal_seq_len * packet_len, num_classes)

    def forward(self, temporal_data, temporal_valid_len):
        temporal_feature = self.temporal_encoder(temporal_data, temporal_valid_len)
        out = F.softmax(self.fc(torch.flatten(temporal_feature, start_dim=1)), dim=-1)
        return out

    def get_attention_weights(self):
        return self.temporal_encoder.attention_weights

    def get_temporal_features(self):
        return self.temporal_encoder.temporal_features


class TrafficScopeContextual(nn.Module):
    def __init__(self, agg_scale_num, agg_points_num, freqs_size,
                 num_heads, num_layers, num_classes, dropout):
        super(TrafficScopeContextual, self).__init__()
        self.model_name = TRAFFIC_SCOPE_CONTEXTUAL
        self.contextual_encoder = ContextualEncoder(agg_scale_num, freqs_size, freqs_size, freqs_size, freqs_size,
                                                    freqs_size, (agg_scale_num * agg_points_num, freqs_size),
                                                    freqs_size, freqs_size * 2,
                                                    num_heads, num_layers, dropout)
        self.fc = nn.Linear(agg_scale_num * agg_points_num * freqs_size, num_classes)

    def forward(self, contextual_data, contextual_segments):
        contextual_feature = self.contextual_encoder(contextual_data, contextual_segments)
        out = F.softmax(self.fc(torch.flatten(contextual_feature, start_dim=1)), dim=-1)
        return out

    def get_attention_weights(self):
        return self.contextual_encoder.attention_weights

    def get_contextual_features(self):
        return self.contextual_encoder.contextual_features
