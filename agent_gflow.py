# File: td3_suite/td3w/agent_gflow.py

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class BehaviorPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = torch.tanh(self.mean_head(x))
        log_std = self.log_std_head(x).clamp(-5, 2)
        return mean, log_std

    def nll(self, state, action):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        var = std**2 + 1e-6
        nll = 0.5 * (((action - mean) ** 2) / var + 2 * log_std + np.log(2 * np.pi))
        return nll.sum(dim=1).mean()

class ActorGaussian(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.max_action * torch.tanh(self.mean_head(x))
        log_std = self.log_std_head(x).clamp(-5, 2)
        return mean, log_std

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa)); q2 = F.relu(self.l5(q2)); q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = self.l3(q1)
        return q1

class GFlow_W2:
    """
    TD3+BC를 gradient flow 관점으로 확장:
      - BehaviorPolicy(가우시안) 회귀
      - Actor(가우시안)과 Behavior 간 W2^2 penalty
      - Actor 엔트로피 보상
      - Critic/타겟 업데이트는 TD3와 동일
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        alpha=2.5,
        w2_weight=1.0,
        entropy_weight=0.01,
    ):
        self.actor = ActorGaussian(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.behavior_policy = BehaviorPolicy(state_dim, action_dim).to(device)
        self.behavior_optimizer = torch.optim.Adam(self.behavior_policy.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.w2_weight = w2_weight
        self.entropy_weight = entropy_weight
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        mean, _ = self.actor.forward(state)
        return mean.cpu().data.numpy().flatten()

    def _train_behavior_policy_one_step(self, replay_buffer, batch_size):
        state, action, _, _, _ = replay_buffer.sample(batch_size)
        loss = self.behavior_policy.nll(state, action)
        self.behavior_optimizer.zero_grad(); loss.backward(); self.behavior_optimizer.step()
        return float(loss.item())

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        # ----- Critic -----
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            target_mean, _ = self.actor_target.forward(next_state)
            next_action = (target_mean + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = reward + not_done * self.discount * torch.min(target_Q1, target_Q2)
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad(); critic_loss.backward(); self.critic_optimizer.step()

        # ----- Behavior (1-step) -----
        behavior_loss = self._train_behavior_policy_one_step(replay_buffer, batch_size)

        metrics = {"critic_loss": float(critic_loss.item()), "behavior_loss": behavior_loss}

        # ----- Actor (delayed) -----
        if self.total_it % self.policy_freq == 0:
            mean_a, log_std_a = self.actor.forward(state)
            Q = self.critic.Q1(state, mean_a)
            lmbda = self.alpha / Q.abs().mean().detach()

            with torch.no_grad():
                mean_b, log_std_b = self.behavior_policy.forward(state)
            std_a = torch.exp(log_std_a); std_b = torch.exp(log_std_b)
            w2 = ((mean_a - mean_b) ** 2).mean() + ((std_a - std_b) ** 2).mean()

            entropy = 0.5 * (2 * log_std_a + np.log(2 * np.pi * np.e)).sum(dim=1).mean()

            actor_loss = self.w2_weight * w2 - lmbda * Q.mean() - self.entropy_weight * entropy

            self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()

            # target soft-update
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            metrics.update({
                "actor_loss": float(actor_loss.item()),
                "lambda": float(lmbda.item()),
                "Q_mean": float(Q.mean().item()),
                "w2_distance": float(w2.item()),
                "entropy": float(entropy.item())
            })

        return metrics

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.behavior_policy.state_dict(), filename + "_behavior")
        torch.save(self.behavior_optimizer.state_dict(), filename + "_behavior_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
        self.behavior_policy.load_state_dict(torch.load(filename + "_behavior"))
        self.behavior_optimizer.load_state_dict(torch.load(filename + "_behavior_optimizer"))