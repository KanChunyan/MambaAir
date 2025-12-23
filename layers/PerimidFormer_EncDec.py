import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations

from layers.SelfAttention_Family import AttentionLayer, FullAttention
from mamba_ssm import Mamba  # 安装 mamba-ssm 库

# --------------------------- mask 构造函数保持不变 ---------------------------
def mask_and_indexes(components_per_level):
    all_comp_num = np.lcm.reduce(components_per_level)
    inclu_rela = np.zeros((len(components_per_level), all_comp_num)).astype(int)

    d = 0
    for i, components in enumerate(components_per_level):
        compent_length = all_comp_num // components
        for j in range(components):
            start_idx = j * compent_length
            end_idx = (j + 1) * compent_length
            inclu_rela[i, start_idx:end_idx] = d
            d += 1

    max_ind = inclu_rela[-1, -1]
    mask_metric = np.zeros((max_ind + 1, max_ind + 1)).astype(int)
    mask_metric[0, 0] = 1

    indexes = []
    indexes_layer_all = []

    for i in range(len(components_per_level)):
        if i == 0:
            continue
        unique_elements_above = np.unique(inclu_rela[i - 1, :])
        unique_elements_down = np.unique(inclu_rela[i, :])
        permutations_result = list(permutations(unique_elements_down, 2))
        idx1, idx2 = np.array(permutations_result).T
        mask_metric[idx1, idx2] = 1

        indexes_layer = []

        for seg_id_down in unique_elements_down:
            mask_metric[seg_id_down, seg_id_down] = 1
            for seg_id_above in unique_elements_above:
                indices = np.where(inclu_rela[i, :] == seg_id_down)[0]
                indices_above = np.where(inclu_rela[i - 1, :] == seg_id_above)[0]
                if (((indices[0] >= indices_above[0]) and (indices[-1] <= indices_above[-1])) or
                    ((indices[0] <= indices_above[-1]) and (indices[-1] >= indices_above[-1])) or
                    ((indices[0] <= indices_above[0]) and (indices[-1] >= indices_above[0]))):
                    mask_metric[seg_id_above, seg_id_down] = 1
                    mask_metric[seg_id_down, seg_id_above] = 1
                    indexes_layer.append([seg_id_above, seg_id_down])
        if i == 1:
            indexes = indexes_layer
            indexes_layer_all.append(indexes_layer)
            continue
        indexes_layer_all.append(indexes_layer)

        indexs_temporary = []
        for index_list in indexes:
            key_value = index_list[-1]
            for sublist in indexes_layer:
                if sublist[0] == key_value:
                    new_list = index_list[:-1] + sublist
                    indexs_temporary.append(new_list)
        indexes = indexs_temporary

    indexes = torch.tensor(np.array(indexes)).unsqueeze(0).unsqueeze(3)
    mask_metric = torch.tensor(1 - mask_metric).bool()
    return mask_metric, indexes


class RegularMask():
    def __init__(self, mask):
        self._mask = mask.unsqueeze(1).unsqueeze(1)

    @property
    def mask(self):
        return self._mask


# --------------------------- 并联 Attention + Mamba 块 ---------------------------
class HybridBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_state=64, expand=2, dropout=0.1):
        super().__init__()
        self.attn = AttentionLayer(
            FullAttention(mask_flag=True, factor=0, attention_dropout=dropout, output_attention=False),
            d_model, n_heads)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, expand=expand)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # x: [B, T, W, D]
        B, T, W, D = x.shape

        # Attention 输入保持4维，Mamba输入需要reshape成3维
        x_attn = self.attn(x, x, x, attn_mask=attn_mask)[0]  # 输出同样为 [B, T, W, D]

        # Mamba 处理 reshape 成 [B, T*W, D]
        x_mamba = x.reshape(B, T * W, D)
        x_mamba = self.mamba(x_mamba)
        x_mamba = x_mamba.reshape(B, T, W, D)

        # 融合并归一化
        x = x + x_attn + x_mamba
        x = self.norm(x)
        return x



class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()
        self.normalize_before = normalize_before
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)
        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual
        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


# --------------------------- 新 EncoderLayer 使用 HybridBlock ---------------------------
class MambaEnhancedEncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, dropout=0.1):
        super().__init__()
        self.hybrid = HybridBlock(d_model=d_model, n_heads=n_head, d_state=64, expand=2, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, x, slf_attn_mask=None, x_upper=None):
        x = self.hybrid(x, attn_mask=RegularMask(slf_attn_mask))
        x = self.ffn(x)
        return x


class Encoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaEnhancedEncoderLayer(
                configs.d_model, configs.d_model, configs.n_heads,
                dropout=configs.dropout
            ) for _ in range(configs.layers)
        ])

    def forward(self, chan_in, components_per_level, task):
        B, L, C, D = chan_in.shape
        chan_in = chan_in.permute(0, 1, 3, 2)
        mask, indexes = mask_and_indexes(components_per_level)
        mask = mask.repeat(B, 1, 1).to(chan_in.device)

        for layer in self.layers:
            chan_in = layer(chan_in, mask)

        enc_out = chan_in.permute(0, 2, 1, 3).reshape(B * C, L, D)
        if task != 'classification':
            num_of_branches = indexes.shape[1]
            indexes = indexes.repeat(enc_out.size(0), 1, 1, enc_out.size(2)).to(enc_out.device)
            indexes = indexes.view(enc_out.size(0), -1, enc_out.size(2))
            enc_out = torch.gather(enc_out, 1, indexes)
            enc_out = enc_out.view(enc_out.size(0), num_of_branches, -1)
        return enc_out.view(B, C, enc_out.shape[-2], enc_out.shape[-1])
