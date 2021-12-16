#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn as nn
import torch.nn.functional as nnfunc
from torch import optim as optim

from habitat.utils import profiling_wrapper
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.dqn.policy import Policy

EPS_DQN = 1e-5


class DQN():
    def __init__(
        self,
        actor_policy: Policy,
        discount_factor: float,
        lr: Optional[float],
        dqn_epoch: int,
        num_mini_batch: int,
        eps: Optional[float] = None,
        use_normalized_advantage: bool = True,
        use_normalized_reward: bool = True,
    ) -> None:

        super().__init__()

        self.actor_policy = actor_policy

        self.dqn_epoch = dqn_epoch
        self.num_mini_batch = num_mini_batch

        self.optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, actor_policy.parameters())),
            lr=lr,
            eps=eps,
        )
        self.device = next(actor_policy.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage
        self.use_normalized_reward = use_normalized_reward


    def get_rewards(self, rollouts: RolloutStorage) -> Tensor:
        rewards = rollouts.buffers["rewards"][:-1]

        if not self.use_normalized_reward:
            return rewards

        return (rewards - rewards.mean()) / (rewards.std() + EPS_DQN)

    def get_advantages(self, rollouts: RolloutStorage) -> Tensor:
        advantages = (
            rollouts.buffers["returns"][:-1]
            - rollouts.buffers["value_preds"][:-1]
        )
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update(self, rollouts: RolloutStorage) -> Tuple[float, float, float]:
        # rewards for evey <num_steps> for all envs
        rewards = self.get_rewards(rollouts)
        advantages = self.get_advantages(rollouts)

        loss_epoch = 0.0

        for _e in range(self.dqn_epoch):
            profiling_wrapper.range_push("DQN.update epoch")

            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for batch in data_generator:
                action_vlaue = self._evaluate_actions(batch["observations"], batch["actions"])

                loss = nnfunc.mse_loss(batch["advantages"], action_vlaue)

                loss = loss.mean()

                self.before_backward(loss)
                loss.backward()
                self.after_backward(loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                loss_epoch += loss.item()

            profiling_wrapper.range_pop()  # DQN.update epoch

        num_updates = self.dqn_epoch * self.num_mini_batch

        loss_epoch /= num_updates

        return loss_epoch

    def _evaluate_actions(self, observations, action):
        return self.actor_policy.get_value(
            observations, action
        )

    def before_backward(self, loss: Tensor) -> None:
        pass

    def after_backward(self, loss: Tensor) -> None:
        pass

    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(
            self.actor_policy.parameters(), self.max_grad_norm
        )

    def after_step(self) -> None:
        pass
