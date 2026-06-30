# 历史观测的 multi-head attention 工具（编码器、位置编码、reshape）。
#
# 4+1 分叉结构：to_time_major 把环境 term-major 扁平历史 reshape 成 [B, H, D]，
# 调用方切出过去帧 [B, H-1, D] 喂 encoder，当前帧 [B, D] 单独喂旁路 l0。
# 原版 fast_td3 的 mask / SOS / valid_length 分支刻意去掉：环境的 observation_manager
# 循环缓冲在 reset 后把首帧复制填满全部 H 槽 -> 不需要 masking。

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_pe(n_positions: int, dim: int) -> torch.Tensor:
    """标准 sinusoidal 位置编码，shape ``[1, n_positions, dim]``。

    给可学位置编码一个带时间结构的初始化（替代零初始化）：step 0 各位置就已可
    区分，attention 不会在第 0 步位置盲。仍是 ``nn.Parameter``，网络后续可自由调整。
    """
    pe = torch.zeros(n_positions, dim)
    position = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    n_cos = pe[:, 1::2].shape[1]
    pe[:, 1::2] = torch.cos(position * div_term[:n_cos])
    return pe.unsqueeze(0)


def to_time_major(obs_flat: torch.Tensor, term_dims: List[int], n_history: int) -> torch.Tensor:
    """把 term-major 扁平历史 reshape 成 time-major ``[B, H, sum(term_dims)]``。

    环境按 term-major 存历史：每个 term 的 n_history 帧连续（旧 -> 新），再把各 term
    拼起来。MHA 要的转置是：n_history 个 token，每个 token 是该时刻完整的一帧观测。
    本函数做的就是这种 per-term 块转置。

    Args:
        obs_flat: ``[B, n_history * sum(term_dims)]`` 的 term-major 扁平历史。
        term_dims: 每个 term 的单步维度，如 ``[3, 3, 3, 29, 29, 29]``。
        n_history: 历史帧数 ``H``。

    Returns:
        ``[B, n_history, sum(term_dims)]`` time-major 张量；``[:, t, :]`` 是第 ``t`` 帧
        的完整观测（``t=0`` 最旧，``t=H-1`` 当前帧）。
    """
    B = obs_flat.shape[0]
    expected = n_history * sum(term_dims)
    if obs_flat.shape[-1] != expected:
        raise ValueError(
            f"to_time_major: expected last dim {expected} "
            f"(n_history={n_history} * sum(term_dims)={sum(term_dims)}), "
            f"got {obs_flat.shape[-1]}"
        )
    chunks = []
    off = 0
    for d in term_dims:
        block = obs_flat[:, off : off + n_history * d]        # 该 term 的 H*d 块
        chunks.append(block.reshape(B, n_history, d))         # [B, H, d]，旧 -> 新
        off += n_history * d
    return torch.cat(chunks, dim=-1)                          # [B, H, sum(term_dims)]


class LinearMHAEncoder(nn.Module):
    """MHA 历史编码器：对过去帧做 self-attention residual block，取最近一帧输出 ``z``。

    term-major -> time-major reshape 已上提到 MHAActor / MHACritic 的 forward 中；
    本编码器直接接收已切出的过去帧 ``[B, H_past, D]``（time-major），不再做 reshape。
    ``n_history`` 设为过去帧数 ``H_past = H - 1``（当前帧由调用方单独切出喂旁路 ``l0``）。

    对应 ``ours/fast_td3_mha_mask.py:LinearMHAEncoder``：去掉 mask / SOS 分支（环境会
    把槽填满），去掉 term-major reshape（调用方负责）。
    输入维度：``[B, H_past, D]``
    输出维度：``[B, hidden_dim]``，取经过 self-attention 后的 t-1 token（last-token pooling）。

    last-token pooling：注意力双向无 mask，``h[:, -1, :]`` 已 attend 过全部过去帧，是
    "带完整历史上下文的最近过去帧 token"，保留 recency；mean pooling 会把位置特异性
    信号平均稀释。``out_norm`` 对残差流做最终 LayerNorm，与调用方旁路的 ``cur_norm``
    对称，保证两路尺度一致。
    """

    def __init__(
        self,
        input_dim: int,       # 单帧观测总维度 D = sum(term_dims)
        n_history: int,       # 传入的过去帧数 H_past = H - 1
        hidden_dim: int,
        nhead: int,
        is_learnable_pos_embedding: bool = True,
        actv=F.elu,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_history = n_history
        self.proj = nn.Linear(input_dim, hidden_dim)          # 单帧线性投影 D -> hidden_dim
        self.actv = actv
        self.pos = (
            nn.Parameter(_sinusoidal_pe(n_history, hidden_dim))
            if is_learnable_pos_embedding
            else None
        )                              # 可学位置编码，sinusoidal 初始化补时序结构；attention 排列不变，需显式补时序
        self.mha = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)          # Pre-LN（attention 前）
        self.out_norm = nn.LayerNorm(hidden_dim)      # 残差流最终归一化，与调用方旁路 cur_norm 对称

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: ``[B, H_past, D]`` 过去帧序列（time-major，已由调用方切出当前帧）。"""
        h = self.proj(x)                # [B, H_past, hidden_dim]
        if self.actv is not None:
            h = self.actv(h)
        if self.pos is not None:
            h = h + self.pos[:, : h.shape[1], :]     # 加位置编码
        h_normed = self.norm(h)                                   # Pre-LN
        h2, _ = self.mha(h_normed, h_normed, h_normed, need_weights=False)  # [B, H_past, hidden_dim]
        h = self.out_norm(h + h2)                                # residual + 最终归一化
        return h[:, -1, :]                                       # [B, hidden_dim]，last-token pooling（t-1，带历史上下文）
