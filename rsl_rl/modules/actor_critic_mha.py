from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.networks.mha import AvgL1Norm, LinearMHAEncoder, to_time_major
from rsl_rl.utils import resolve_nn_activation


class MHAActor(nn.Module):
    """以 MHA 历史 embedding ``z`` 为条件的 actor 主干。

    4+1 分叉：对 term-major 扁平历史先 ``to_time_major`` 成 ``[B, H, D]``（time-major），
    切出 **当前帧** ``[B, D]`` 喂旁路 ``l0``，**过去帧** ``[B, H-1, D]`` 喂
    ``LinearMHAEncoder``。两路 cat 后进 trunk MLP 出动作均值（无界；rsl_rl 保持
    无界高斯，动作由环境 clip）。

    对应 ``ours/fast_td3_mha_mask.py:Actor`` 的分工：``l0`` 只压当前状态，
    ``z`` 只编过去时序——原版 obs 和 z 是两路独立输入，这里统一从 term-major
    扁平 obs 内部切分。
    """

    def __init__(self, term_dims, num_actions, n_history, enc_hidden,
                 nheads, pos_emb, hidden_dims, activation):
        super().__init__()
        actv = resolve_nn_activation(activation)
        self.n_history = n_history                  # 环境扁平帧数 H（如 5）
        self.term_dims = list(term_dims)            # 各 term 单步维度
        self.single_dim = int(sum(self.term_dims))  # 单帧观测总维度 D
        self.in_features = n_history * self.single_dim

        n_past = n_history - 1                      # 过去帧数
        self.encoder = LinearMHAEncoder(
            input_dim=self.single_dim, n_history=n_past, hidden_dim=enc_hidden,
            nhead=nheads, is_learnable_pos_embedding=pos_emb, actv=actv,
        )                                          # 过去帧编码器 -> z [B, enc_hidden]
        self.l0 = nn.Linear(self.single_dim, enc_hidden)   # 当前帧旁路投影 D -> enc_hidden
        self.trunk = MLP(2 * enc_hidden, num_actions, hidden_dims, activation)

    def __getitem__(self, index: int):
        # ponytail: IsaacLab's ONNX exporter reads actor[0].in_features for dummy obs.
        if index == 0:
            return self
        raise IndexError(index)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B, H*D] term-major 扁平历史
        x = to_time_major(obs, self.term_dims, self.n_history)  # [B, H, D]
        cur = x[:, -1, :]        # 当前帧  [B, D]
        past = x[:, :-1, :]      # 过去帧  [B, H-1, D]
        z = self.encoder(past)                                   # 历史注意力摘要
        h = AvgL1Norm(self.l0(cur))                              # 当前帧旁路，自归一化
        return self.trunk(torch.cat([h, z], dim=-1))             # 拼两路进 MLP -> 动作均值


class MHACritic(nn.Module):
    """以 MHA 历史 embedding 为条件的 value 主干（4+1 分叉，与 MHAActor 结构一致，输出 1 维）。"""

    def __init__(self, term_dims, n_history, enc_hidden, nheads, pos_emb,
                 hidden_dims, activation):
        super().__init__()
        actv = resolve_nn_activation(activation)
        self.n_history = n_history
        self.term_dims = list(term_dims)
        self.single_dim = int(sum(self.term_dims))

        n_past = n_history - 1
        self.encoder = LinearMHAEncoder(
            input_dim=self.single_dim, n_history=n_past, hidden_dim=enc_hidden,
            nhead=nheads, is_learnable_pos_embedding=pos_emb, actv=actv,
        )
        self.l0 = nn.Linear(self.single_dim, enc_hidden)   # 当前帧旁路投影
        self.trunk = MLP(2 * enc_hidden, 1, hidden_dims, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = to_time_major(obs, self.term_dims, self.n_history)  # [B, H, D]
        cur = x[:, -1, :]        # 当前帧  [B, D]
        past = x[:, :-1, :]      # 过去帧  [B, H-1, D]
        z = self.encoder(past)
        h = AvgL1Norm(self.l0(cur))
        return self.trunk(torch.cat([h, z], dim=-1))


class ActorCriticMHA(ActorCritic):
    """带 MHA 历史编码器的 ActorCritic：actor 必带 MHA，critic 可选（``use_critic_mha``）。

    内部采用 4+1 分叉：当前帧走旁路 ``l0``，过去帧走 ``LinearMHAEncoder``，两路 cat
    后进 trunk MLP。``ActorCritic`` 的 drop-in 替换：``__init__`` 接口一致，外加 MHA
    参数。``act`` / ``evaluate`` / ``update_distribution`` 等全部继承不改——encoder 折进
    ``self.actor`` / ``self.critic``，基类方法调 ``self.actor(obs)`` 照常工作。

    ``actor_term_dims`` / ``critic_term_dims`` 必须与环境 obs term 布局一致。G1-29dof velocity：
      actor  = ``[3, 3, 3, 29, 29, 29]``（单步 96，5 帧扁平 480），
      critic = ``[3, 3, 3, 3, 29, 29, 29]``（单步 99，5 帧扁平 495）。
    """

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] = [256, 256, 256],
        critic_hidden_dims: list[int] = [256, 256, 256],
        activation: str = "elu",
        n_history: int = 5,
        nheads: int = 4,
        encoder_hidden_dim: int | None = 64,
        is_learnable_pos_embedding: bool = True,
        use_critic_mha: bool = False,
        actor_term_dims: list[int] | None = None,
        critic_term_dims: list[int] | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticMHA.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        # 先走基类 __init__：建归一化器、std、分布占位，以及普通 MLP 的 self.actor /
        # self.critic。下面用 MHA 版本覆盖掉它们。
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
        )

        if actor_term_dims is None or critic_term_dims is None:
            raise ValueError(
                "ActorCriticMHA requires actor_term_dims and critic_term_dims "
                "(per-term single-step dims of the env's obs groups)."
            )

        enc_hidden_a = encoder_hidden_dim if encoder_hidden_dim is not None else actor_hidden_dims[0] // 2
        self.actor = MHAActor(
            term_dims=actor_term_dims, num_actions=num_actions,
            n_history=n_history, enc_hidden=enc_hidden_a,
            nheads=nheads, pos_emb=is_learnable_pos_embedding,
            hidden_dims=actor_hidden_dims, activation=activation,
        )

        if use_critic_mha:
            enc_hidden_c = encoder_hidden_dim if encoder_hidden_dim is not None else critic_hidden_dims[0] // 2
            self.critic = MHACritic(
                term_dims=critic_term_dims, n_history=n_history,
                enc_hidden=enc_hidden_c, nheads=nheads, pos_emb=is_learnable_pos_embedding,
                hidden_dims=critic_hidden_dims, activation=activation,
            )
        # 否则保留基类的普通 MLP self.critic（critic 不走 MHA）。

        print(f"Actor MHA: {self.actor}")
        print(f"Critic {'MHA' if use_critic_mha else 'MLP'}: {self.critic}")
