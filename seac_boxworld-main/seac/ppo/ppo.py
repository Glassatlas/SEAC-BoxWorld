import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from gym.spaces import flatdim

from model import Policy, NNBase, DRRLBase
from storage import RolloutStorage


@dataclass
class AlgoConfig:
    lr: float = 3e-4
    adam_eps: float = 0.001
    gamma: float = 0.99
    use_gae: bool = False
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    clip_param: float = 0.2
    use_clipped_value_loss: bool = True
    ppo_epoch: int = 10
    num_mini_batch: int = 10

    use_proper_time_limits: bool = True
    use_recurrent_policy: bool = False
    use_linear_lr_decay: bool = False

    seac_coef: float = 1.0

    num_processes: int = 4
    num_steps: int = 5

    device: str = "cpu"
    base: Optional[NNBase] = DRRLBase


@dataclass
class NetConfig:
    n_f_conv1: int = 12
    n_f_conv2: int = 24
    att_emb_size: int = 64
    n_heads: int = 2
    n_att_stack: int = 2
    n_fc_layers: int = 4
    pad: bool = True  # padding will maintain size of state space
    baseline_mode: bool = False  # will replace attentional module with several convolutional layers to create baseline module
    n_baseMods: int = 3  # 3and 6 are default in paper


class PPO:
    def __init__(
            self,
            agent_id: int,
            obs_space,
            action_space,
            algo_config: AlgoConfig,
            net_config: NetConfig,
            # actor_critic,
            # clip_param,
            # ppo_epoch,
            # num_mini_batch,
            # value_loss_coef,
            # entropy_coef,
            # lr=None,
            # eps=None,
            # max_grad_norm=None,
            # use_clipped_value_loss=True
    ):
        self.agent_id = agent_id
        self.obs_size = flatdim(obs_space)
        self.action_size = flatdim(action_space)
        self.obs_space = obs_space
        self.action_space = action_space
        self.algo_config = algo_config
        self.net_config = net_config

        # self.actor_critic = actor_critic
        # self.clip_param = clip_param
        # self.ppo_epoch = ppo_epoch
        # self.num_mini_batch = num_mini_batch
        # self.value_loss_coef = value_loss_coef
        # self.entropy_coef = entropy_coef
        # self.max_grad_norm = max_grad_norm
        # self.use_clipped_value_loss = use_clipped_value_loss
        # self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        base_kwargs = net_config.__dict__
        # base_kwargs = {}
        base_kwargs["recurrent"] = algo_config.use_recurrent_policy

        self.model = Policy(
            obs_space=obs_space,
            action_space=action_space,
            base=algo_config.base,
            base_kwargs=base_kwargs,
        )

        self.storage = RolloutStorage(
            obs_space,
            action_space,
            self.model.recurrent_hidden_state_size,
            algo_config.num_steps,
            algo_config.num_processes,
        )

        self.model.to(algo_config.device)
        self.optimizer = optim.Adam(self.model.parameters(), algo_config.lr, eps=algo_config.adam_eps)

        # self.intr_stats = RunningStats()
        self.saveables = {
            "model": self.model,
            "optimizer": self.optimizer,
        }

    def save(self, path):
        torch.save(self.saveables, os.path.join(path, "models.pt"))

    def restore(self, path):
        checkpoint = torch.load(os.path.join(path, "models.pt"))
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    def compute_returns(self, use_gae, gamma, gae_lambda, use_proper_time_limits, **kwargs):
        with torch.no_grad():
            next_value = self.model.get_value(
                self.storage.obs[-1],
                self.storage.recurrent_hidden_states[-1],
                self.storage.masks[-1],
            ).detach()

        self.storage.compute_returns(
            next_value, use_gae, gamma, gae_lambda, use_proper_time_limits,
        )

    def update(
            self,
            rollouts,
            storages,
            value_loss_coef,
            entropy_coef,
            seac_coef,
            max_grad_norm,
            device,
            **kwargs
    ):
        obs_shape = self.storage.obs.size()[2:]
        action_shape = self.storage.actions.size()[-1]
        num_steps, num_processes, _ = self.storage.rewards.size()

        values, action_log_probs, dist_entropy, _ = self.model.evaluate_actions(
            self.storage.obs[:-1].view(-1, *obs_shape),
            self.storage.recurrent_hidden_states[0].view(-1, self.model.recurrent_hidden_state_size),
            self.storage.masks[:-1].view(-1, 1),
            self.storage.actions.view(-1, action_shape),
        )

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = self.storage.returns[:-1] - values

        # advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        policy_loss_epoch = 0
        dist_entropy_epoch = 0

        seac_value_loss_epoch = 0
        seac_policy_loss_epoch = 0

        for e in range(self.algo_config.ppo_epoch):
            if self.model.is_recurrent:
                data_generator = self.storage.recurrent_generator(advantages, self.algo_config.num_mini_batch)
            else:
                data_generator = self.storage.feed_forward_generator(advantages, self.algo_config.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.model.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    masks_batch,
                    actions_batch
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(
                    ratio, 1.0 - self.algo_config.clip_param, 1.0 + self.algo_config.clip_param
                ) * adv_targ
                policy_loss = -torch.min(surr1, surr2).mean()

                if self.algo_config.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                        -self.algo_config.clip_param, self.algo_config.clip_param
                    )
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # calculate prediction loss for the OTHER actor
                other_agent_ids = [x for x in range(len(storages)) if x != self.agent_id]
                seac_policy_loss = 0
                seac_value_loss = 0
                for oid in other_agent_ids:
                    other_values, logp, _, _ = self.model.evaluate_actions(
                        storages[oid].obs[:-1].view(-1, *obs_shape),
                        storages[oid]
                            .recurrent_hidden_states[0]
                            .view(-1, self.model.recurrent_hidden_state_size),
                        storages[oid].masks[:-1].view(-1, 1),
                        storages[oid].actions.view(-1, action_shape),
                    )
                    other_values = other_values.view(num_steps, num_processes, 1)
                    logp = logp.view(num_steps, num_processes, 1)
                    other_advantage = (storages[oid].returns[:-1] - other_values)  # or storages[oid].rewards

                    importance_sampling = (logp.exp() / (storages[oid].action_log_probs.exp() + 1e-7)).detach()
                    # importance_sampling = 1.0
                    seac_value_loss += (importance_sampling * other_advantage.pow(2)).mean()
                    seac_policy_loss += (-importance_sampling * logp * other_advantage.detach()).mean()

                self.optimizer.zero_grad()
                (
                        policy_loss
                        + self.algo_config.value_loss_coef * value_loss
                        - self.algo_config.entropy_coef * dist_entropy
                        + self.algo_config.seac_coef * seac_policy_loss
                        + self.algo_config.seac_coef * self.algo_config.value_loss_coef * seac_value_loss
                ).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.algo_config.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                policy_loss_epoch += policy_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.algo_config.ppo_epoch * self.algo_config.num_mini_batch

        value_loss_epoch /= num_updates
        policy_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        # return value_loss_epoch, policy_loss_epoch, dist_entropy_epoch

        return {
            "policy_loss": policy_loss_epoch,
            "value_loss": value_loss_coef * value_loss_epoch,
            "dist_entropy": entropy_coef * dist_entropy_epoch,
            "importance_sampling": importance_sampling.mean().item(),
            "seac_policy_loss": seac_coef * seac_policy_loss_epoch,
            "seac_value_loss": seac_coef * value_loss_coef * seac_value_loss_epoch,
        }
