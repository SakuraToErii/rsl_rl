from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.networks.mha import AvgL1Norm, LinearMHAEncoder
from rsl_rl.utils import resolve_nn_activation


class MHAActor(nn.Module):
    """Actor trunk conditioned on an MHA history embedding ``z``.

    Mirrors ``ours/fast_td3_mha_mask.py:Actor``: ``cat([AvgL1Norm(l0(obs_flat)), z])``
    then an MLP head. The encoder consumes the term-major flat obs directly and
    does the time-major reshape internally. Outputs the action mean (unbounded;
    rsl_rl keeps an unbounded Gaussian and the env clips actions).
    """

    def __init__(self, num_obs, num_actions, n_history, term_dims, enc_hidden,
                 nheads, pos_emb, hidden_dims, activation):
        super().__init__()
        actv = resolve_nn_activation(activation)
        self.encoder = LinearMHAEncoder(
            term_dims=term_dims, n_history=n_history, hidden_dim=enc_hidden,
            nhead=nheads, is_learnable_pos_embedding=pos_emb, actv=actv,
        )
        self.l0 = nn.Linear(num_obs, enc_hidden)
        self.trunk = MLP(2 * enc_hidden, num_actions, hidden_dims, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs)
        h = AvgL1Norm(self.l0(obs))
        return self.trunk(torch.cat([h, z], dim=-1))


class MHACritic(nn.Module):
    """Value trunk conditioned on an MHA history embedding (same shape as ``MHAActor``)."""

    def __init__(self, num_obs, n_history, term_dims, enc_hidden, nheads, pos_emb,
                 hidden_dims, activation):
        super().__init__()
        actv = resolve_nn_activation(activation)
        self.encoder = LinearMHAEncoder(
            term_dims=term_dims, n_history=n_history, hidden_dim=enc_hidden,
            nhead=nheads, is_learnable_pos_embedding=pos_emb, actv=actv,
        )
        self.l0 = nn.Linear(num_obs, enc_hidden)
        self.trunk = MLP(2 * enc_hidden, 1, hidden_dims, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs)
        h = AvgL1Norm(self.l0(obs))
        return self.trunk(torch.cat([h, z], dim=-1))


class ActorCriticMHA(ActorCritic):
    """``ActorCritic`` with an MHA history encoder on the actor (always) and
    optionally on the critic (``use_critic_mha``).

    Drop-in replacement for ``ActorCritic``: same ``__init__`` surface plus the
    MHA parameters. ``act`` / ``evaluate`` / ``update_distribution`` / ... are
    inherited unchanged -- the encoder is folded inside ``self.actor`` /
    ``self.critic``, so the base methods (which call ``self.actor(obs)`` /
    ``self.critic(obs)`` on the normalized flat obs) work as-is.

    ``actor_term_dims`` / ``critic_term_dims`` must match the env's obs term
    layout. For G1-29dof velocity:
      actor  = ``[3, 3, 3, 29, 29, 29]`` (96 single-step x 5 = 480),
      critic = ``[3, 3, 3, 3, 29, 29, 29]`` (99 single-step x 5 = 495).
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
        nheads: int = 8,
        encoder_hidden_dim: int | None = None,
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
        # Build normalizers, std and the distribution placeholder via the base
        # class. The base also builds plain-MLP self.actor / self.critic, which
        # we overwrite below with the MHA variants.
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

        # Base computed these locally but did not store them; recompute here.
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])

        enc_hidden_a = encoder_hidden_dim if encoder_hidden_dim is not None else actor_hidden_dims[0] // 2
        self.actor = MHAActor(
            num_obs=num_actor_obs, num_actions=num_actions,
            n_history=n_history, term_dims=actor_term_dims, enc_hidden=enc_hidden_a,
            nheads=nheads, pos_emb=is_learnable_pos_embedding,
            hidden_dims=actor_hidden_dims, activation=activation,
        )

        if use_critic_mha:
            enc_hidden_c = encoder_hidden_dim if encoder_hidden_dim is not None else critic_hidden_dims[0] // 2
            self.critic = MHACritic(
                num_obs=num_critic_obs, n_history=n_history, term_dims=critic_term_dims,
                enc_hidden=enc_hidden_c, nheads=nheads, pos_emb=is_learnable_pos_embedding,
                hidden_dims=critic_hidden_dims, activation=activation,
            )
        # else: keep the base plain-MLP self.critic.

        print(f"Actor MHA: {self.actor}")
        print(f"Critic {'MHA' if use_critic_mha else 'MLP'}: {self.critic}")