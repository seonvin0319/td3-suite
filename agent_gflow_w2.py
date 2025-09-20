# File: td3_suite/td3w/agent_gflow_w2.py

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- utils -----------------
def gaussian_nll(x: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    # x, mean, log_std: [B, A]
    var = (log_std.exp() ** 2) + 1e-6
    nll = 0.5 * (((x - mean) ** 2) / var + 2 * log_std + math.log(2 * math.pi))  # [B, A]
    return nll.sum(dim=1).mean()

def diag_gauss_entropy(log_std: torch.Tensor) -> torch.Tensor:
    # H = 0.5 * sum(log(2πeσ^2))  (나이브: tanh 보정 없음)
    return 0.5 * (2 * log_std + math.log(2 * math.pi * math.e)).sum(dim=1).mean()

class EMA:
    def __init__(self, beta=0.995, init=10.0):
        self.beta = beta
        self.v = init
    def update(self, x: float):
        self.v = self.beta * self.v + (1 - self.beta) * float(x)
        return self.v

# ----------------- networks -----------------
class BehaviorPolicy(nn.Module):
    """
    액션공간 Gaussian: a ~ N(mean_b, diag(std_b^2))
    데이터셋 (s,a)에 대해 NLL 최소화. mean_b는 [-max_action, max_action].
    """
    def __init__(self, state_dim, action_dim, max_action, log_std_min=-3.0, log_std_max=-1.0):
        super().__init__()
        self.max_action = float(max_action)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state)); x = F.relu(self.l2(x))
        mean = self.max_action * torch.tanh(self.mean_head(x))
        log_std = self.log_std_head(x).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def nll(self, state, action):
        mean, log_std = self.forward(state)
        return gaussian_nll(action, mean, log_std)

class ActorGaussian(nn.Module):
    """
    액션공간 Gaussian: a ~ N(mean, diag(std^2)), mean은 [-max_action, max_action]
    """
    def __init__(self, state_dim, action_dim, max_action, log_std_min=-3.0, log_std_max=-1.0):
        super().__init__()
        self.max_action = float(max_action)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state)); x = F.relu(self.l2(x))
        mean = self.max_action * torch.tanh(self.mean_head(x))
        log_std = self.log_std_head(x).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    @torch.no_grad()
    def sample_no_grad(self, state, use_mean=False):
        mean, log_std = self.forward(state)
        if use_mean:
            return mean
        std = log_std.exp()
        eps = torch.randn_like(std)
        a = mean + std * eps
        # 타깃/행동 샘플은 범위 맞춰줌
        return a.clamp(-self.max_action, self.max_action)

    def rsample_squashed(self, state, use_fixed_std=None):
        """
        학습용 reparameterized 샘플. clamp를 쓰지 않고 tanh-squash로 그라디언트 보존.
        a = max * tanh( (mean + std*eps) / max )
        """
        mean, log_std = self.forward(state)
        std = (torch.full_like(log_std, use_fixed_std) if use_fixed_std is not None else log_std.exp())
        eps = torch.randn_like(std)
        raw = mean + std * eps
        a = self.max_action * torch.tanh(raw / self.max_action)
        return a, mean, log_std  # raw는 안 씀

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

# ----------------- agent (W2) -----------------
class TD3_GFlow_W2:
    """
    TD3 + W2-regularized stochastic policy (액션공간 Gaussian):
      - Behavior: 액션공간 Gaussian 회귀(NLL)
      - Actor: W2^2(mean,std) vs Behavior + (옵션) 엔트로피
      - Critic: 타깃은 mean(기본) 또는 K-샘플 평균
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_freq=2,
        alpha=2.5,               # λ = alpha / EMA(|Q|)
        w2_weight=1.0,           # (외부 스케줄)
        w2_std_weight=0.5,       # std 항 비중 낮춤(기본 0.5)
        entropy_weight=0.0,      # 기본 0 (나중에 0.005~0.02 시도)
        # 타깃 백업
        target_use_mean=True,    # 기본: 타깃은 mean만 (매우 안정)
        target_K=2,              # >1: 타깃 평균으로 분산 더 감소
        # 초기 안정화
        fixed_actor_std=0.2,     # 초기 고정 std
        freeze_std_steps=20000,  # 이 스텝까지는 고정 std 사용
        # 옵티마이저 & 범위
        lr_actor=3e-4, lr_critic=3e-4, lr_behavior=3e-4,
        log_std_min=-3.0, log_std_max=-1.0,
        grad_clip_norm=None      # 예: 5.0 넣으면 clip
    ):
        self.max_action = float(max_action)
        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.w2_weight = w2_weight
        self.w2_std_weight = w2_std_weight
        self.entropy_weight = entropy_weight
        self.target_use_mean = target_use_mean
        self.target_K = target_K

        self.fixed_actor_std = fixed_actor_std
        self.freeze_std_steps = freeze_std_steps
        self.grad_clip_norm = grad_clip_norm

        self.actor = ActorGaussian(state_dim, action_dim, max_action, log_std_min, log_std_max).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.behavior_policy = BehaviorPolicy(state_dim, action_dim, max_action, log_std_min, log_std_max).to(device)
        self.behavior_optimizer = torch.optim.Adam(self.behavior_policy.parameters(), lr=lr_behavior)

        self.total_it = 0
        self.q_abs_ema = EMA(beta=0.995, init=10.0)  # λ 안정화

    # -------- acting --------
    def select_action(self, state, deterministic: bool = True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if deterministic:
            mean, _ = self.actor.forward(state)
            return mean.cpu().numpy().flatten()
        else:
            a = self.actor.sample_no_grad(state, use_mean=False)
            return a.cpu().numpy().flatten()

    # -------- behavior one-step --------
    def _train_behavior_policy_one_step(self, replay_buffer, batch_size):
        state, action, _, _, _ = replay_buffer.sample(batch_size)
        loss = self.behavior_policy.nll(state, action)
        self.behavior_optimizer.zero_grad()
        loss.backward()
        self.behavior_optimizer.step()
        return float(loss.item())

    # -------- train --------
    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # ----- Critic -----
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        with torch.no_grad():
            if self.target_K <= 1:
                next_action = self.actor_target.sample_no_grad(next_state, use_mean=self.target_use_mean)
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = reward + not_done * self.discount * torch.min(target_Q1, target_Q2)
            else:
                q1_list, q2_list = [], []
                for _ in range(self.target_K):
                    na = self.actor_target.sample_no_grad(next_state, use_mean=self.target_use_mean)
                    q1k, q2k = self.critic_target(next_state, na)
                    q1_list.append(q1k); q2_list.append(q2k)
                q1_avg = torch.stack(q1_list, dim=0).mean(0)
                q2_avg = torch.stack(q2_list, dim=0).mean(0)
                target_Q = reward + not_done * self.discount * torch.min(q1_avg, q2_avg)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()

        # ----- Behavior (1-step) -----
        behavior_loss = self._train_behavior_policy_one_step(replay_buffer, batch_size)

        metrics = {"critic_loss": float(critic_loss.item()), "behavior_loss": behavior_loss}

        # ----- Actor (delayed) -----
        if self.total_it % self.policy_freq == 0:
            # 초반엔 고정 std로, 이후엔 학습 std로 rsample (tanh-squash)
            use_fixed_std = self.fixed_actor_std if self.total_it <= self.freeze_std_steps else None
            a_samp, mean_a, logstd_a = self.actor.rsample_squashed(state, use_fixed_std=use_fixed_std)

            Q = self.critic.Q1(state, a_samp)
            q_abs = Q.abs().mean().item()
            q_scale = max(self.q_abs_ema.update(q_abs), 1e-3)
            lmbda = self.alpha / q_scale

            with torch.no_grad():
                mean_b, logstd_b = self.behavior_policy.forward(state)

            std_a = logstd_a.exp(); std_b = logstd_b.exp()
            w2_mean = (mean_a - mean_b).pow(2).mean()
            w2_std  = (std_a - std_b).pow(2).mean()
            w2 = w2_mean + self.w2_std_weight * w2_std

            if self.entropy_weight > 0.0:
                entropy = diag_gauss_entropy(logstd_a)
            else:
                entropy = torch.tensor(0.0, device=state.device)

            actor_loss = self.w2_weight * w2 - lmbda * Q.mean() - self.entropy_weight * entropy

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
            self.actor_optimizer.step()

            # target soft-update
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            metrics.update({
                "actor_loss": float(actor_loss.item()),
                "lambda": float(lmbda),
                "Q_mean": float(Q.mean().item()),
                "w2_distance": float(w2.item()),
                "w2_mean": float(w2_mean.item()),
                "w2_std": float(w2_std.item()),
                "entropy": float(entropy.item()) if isinstance(entropy, torch.Tensor) else float(entropy),
                "q_abs_ema": float(q_scale)
            })

        return metrics

    # -------- io --------
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.behavior_policy.state_dict(), filename + "_behavior")
        torch.save(self.behavior_optimizer.state_dict(), filename + "_behavior_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=device))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=device))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=device))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=device))
        self.actor_target = copy.deepcopy(self.actor)

        self.behavior_policy.load_state_dict(torch.load(filename + "_behavior", map_location=device))
        self.behavior_optimizer.load_state_dict(torch.load(filename + "_behavior_optimizer", map_location=device))
