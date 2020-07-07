#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import numpy as np

EPS_SAC = 1e-5


class SAC(nn.Module):
    def __init__(
        self,
        actor_critic,
        clip_param,
        sac_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        dim_actions,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        use_normalized_advantage=True,
        alpha = None,
        gamma = 0.99,
    ):

        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.sac_epoch = sac_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.dim_actions = dim_actions

        self.actor_optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, actor_critic.parameters())),
            lr=lr,
            eps=eps,
        )

        net = actor_critic.get_net()

        self.device = next(actor_critic.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage
        self.critic_local_1 = CriticHeadSAC(net.output_size, self.dim_actions)
        self.critic_local_2 = CriticHeadSAC(net.output_size, self.dim_actions)

        self.critic_target_1 =  CriticHeadSAC(net.output_size, self.dim_actions)
        self.critic_target_2 =  CriticHeadSAC(net.output_size, self.dim_actions)

        self.copy_model(self.critic_local_1, self.critic_target_1)
        self.copy_model(self.critic_local_2, self.critic_target_2)

        

        local_1_params = list(filter(lambda p: p.requires_grad, self.critic_local_1.parameters())) #+ list(filter(lambda p: p.requires_grad, net.parameters()))
        local_2_params = list(filter(lambda p: p.requires_grad, self.critic_local_2.parameters())) #+ list(filter(lambda p: p.requires_grad, net.parameters()))
        self.local_1_optimizer = optim.Adam(
            local_1_params,
            lr=lr,
            eps=eps,
        )
        self.local_2_optimizer = optim.Adam(
            local_2_params,
            lr=lr,
            eps=eps,
        )
        self.mse_loss = nn.MSELoss()

        self.target_entropy = -np.log((1.0 / self.dim_actions)) * 0.98
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = optim.SGD([self.log_alpha], lr)

        self.gamma = gamma

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_SAC)

    def update(self, rollouts):
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.sac_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            #pdb.set_trace()
            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                # Reshape to do in a single forward pass for all steps
                (
                    features,
                    action_probs,
                    action_log_probs,
                    dist_entropy,
                    _,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                )

                q_target_1 = self.critic_target_1(features)
                q_target_2 = self.critic_target_2(features)
                q_next_target = action_probs*(torch.min(q_target_1, q_target_2) - self.alpha*action_log_probs)
                q_next_target = q_next_target.mean(dim=1).unsqueeze(-1)
                next_q_value = return_batch + self.gamma*q_next_target

                q_local_1 = self.critic_local_1(features)
                q_local_2 = self.critic_local_2(features)

                loss_local_1 = self.mse_loss(q_local_1.gather(1, prev_actions_batch), next_q_value).mean()
                loss_local_2 = self.mse_loss(q_local_2.gather(1, prev_actions_batch), next_q_value).mean()

                
                min_q_local = torch.min(q_local_1, q_local_1)
                loss_policy = action_probs*(-min_q_local + self.alpha*action_log_probs)
                loss_policy = loss_policy.mean()

                loss_alpha = -(self.log_alpha*(action_log_probs + self.target_entropy).detach()).mean()


                ##TO:DO make a function to call optimizers given params
                self.actor_optimizer.zero_grad()
                loss_policy.backward(retain_graph= True)
                self.before_step(self.actor_critic)
                self.actor_optimizer.step()

                self.local_1_optimizer.zero_grad()
                loss_local_1.backward(retain_graph= True)
                self.before_step(self.critic_local_1)
                self.local_1_optimizer.step()

                self.local_2_optimizer.zero_grad()
                loss_local_2.backward(retain_graph= True)
                self.before_step(self.critic_local_2)
                self.local_2_optimizer.step()

                self.alpha_optim.zero_grad()
                loss_alpha.backward(retain_graph= True)
                nn.utils.clip_grad_norm_(self.log_alpha, self.max_grad_norm)
                self.alpha_optim.step()
                self.alpha = self.log_alpha.exp()

                value_loss_epoch += loss_local_1.item()
                action_loss_epoch += loss_policy.item()
                dist_entropy_epoch += dist_entropy.item()

                self.soft_update_of_target_network(self.critic_local_1, self.critic_target_1)
                self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2)

        num_updates = self.sac_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step(self, network):
        nn.utils.clip_grad_norm_(
            network.parameters(), self.max_grad_norm
        )

    def apply_update(self):
        pass


    def after_step(self):
        pass

    def soft_update_of_target_network(self, local_model, target_model, polyak=0.005):
            """Updates the target network in the direction of the local network but by taking a step size
            less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(polyak*local_param.data + (1.0-polyak)*target_param.data)



    def copy_model(self, from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())


class CriticHeadSAC(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)
