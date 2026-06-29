# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic
from .actor_critic_mha import ActorCriticMHA  # MHA 历史编码器变体（本次移植新增）；导出后 eval(class_name) 可解析
from .actor_critic_recurrent import ActorCriticRecurrent
from .rnd import *
from .student_teacher import StudentTeacher
from .student_teacher_recurrent import StudentTeacherRecurrent
from .symmetry import *

__all__ = [
    "ActorCritic",
    "ActorCriticMHA",
    "ActorCriticRecurrent",
    "StudentTeacher",
    "StudentTeacherRecurrent",
]