# MHA history encoder for term-major flat observations.
#
# Ported from ours/fast_td3_mha_mask.py (LinearMHAEncoder) and adapted to
# unitree_rl_lab's term-major flattened history layout. The mask / SOS /
# valid_length branches from the original are removed on purpose: the env's
# observation_manager circular buffer fills all H slots (first-frame repeat
# after reset), so the history the policy sees is always full -> no masking.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def AvgL1Norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Self-normalizing activation: divide by the mean of absolute values."""
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def to_time_major(obs_flat: torch.Tensor, term_dims, n_history: int) -> torch.Tensor:
    """Reshape a term-major flattened history into time-major ``[B, H, sum(term_dims)]``.

    The env lays out history term-major: each term's ``n_history`` frames are
    contiguous (old -> new), then the terms are concatenated. Multi-head
    attention needs the transpose: ``n_history`` tokens, each token being the
    full single-step observation at that timestep. This function performs that
    per-term block transpose.

    Args:
        obs_flat: ``[B, n_history * sum(term_dims)]`` term-major flat history.
        term_dims: per-term single-step dims, e.g. ``[3, 3, 3, 29, 29, 29]``.
        n_history: number of history frames ``H``.

    Returns:
        ``[B, n_history, sum(term_dims)]`` time-major tensor; ``[:, t, :]`` is
        the full observation at frame ``t`` (``t=0`` oldest, ``t=H-1`` current).
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
        block = obs_flat[:, off : off + n_history * d]        # this term's H*d block
        chunks.append(block.reshape(B, n_history, d))         # [B, H, d], old -> new
        off += n_history * d
    return torch.cat(chunks, dim=-1)                          # [B, H, sum(term_dims)]


class LinearMHAEncoder(nn.Module):
    """Multi-head attention encoder over a fixed-length observation history.

    Input is the term-major flat history straight from the env; internally it
    is reshaped to time-major ``[B, H, D]``, projected, optionally given a
    learnable positional embedding, run through self-attention, and
    mean-pooled (all ``H`` slots are valid here) to a fixed ``[B, hidden_dim]``
    embedding ``z``.

    Mirrors ``ours/fast_td3_mha_mask.py:LinearMHAEncoder`` with the mask / SOS
    branches removed (env fills all slots) and the term-major -> time-major
    reshape added (env stores history term-major flat).
    """

    def __init__(
        self,
        term_dims,
        n_history: int,
        hidden_dim: int,
        nhead: int,
        is_learnable_pos_embedding: bool = True,
        actv=F.elu,
    ):
        super().__init__()
        self.term_dims = list(term_dims)
        self.n_history = n_history
        self.single_dim = int(sum(self.term_dims))
        self.proj = nn.Linear(self.single_dim, hidden_dim)
        self.actv = actv
        self.pos = (
            nn.Parameter(torch.zeros(1, n_history, hidden_dim))
            if is_learnable_pos_embedding
            else None
        )
        self.mha = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True)

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        x = to_time_major(obs_flat, self.term_dims, self.n_history)   # [B, H, single_dim]
        h = self.proj(x)
        if self.actv is not None:
            h = self.actv(h)
        if self.pos is not None:
            h = h + self.pos[:, : h.shape[1], :]
        h2, _ = self.mha(h, h, h, need_weights=False)                 # [B, H, hidden_dim]
        return h2.mean(dim=1)                                         # [B, hidden_dim]