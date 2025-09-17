import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
EPS = 1e-8

def atanh_clip(x, eps=1e-6):
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

# ----------------- Nets -----------------
class BehaviorPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state)); x = F.relu(self.l2(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def nll(self, state, action):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        var = std**2 + 1e-6
        nll = 0.5 * (((action - mean) ** 2) / var + 2 * log_std + np.log(2*np.pi))
        return nll.sum(dim=1).mean()


class ActorGaussian(nn.Module):
    """Deterministic mean (for action) + stochastic head for entropy regularization."""
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state)); x = F.relu(self.l2(x))
        mean = self.max_action * torch.tanh(self.mean_head(x))
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std  # mean used for action; log_std for entropy term

    def action(self, mean):
        return self.max_action * torch.tanh(mean)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa)); q2 = F.relu(self.l5(q2)); q2 = self.l6(q2)
        return q1, q2

    def Q1(self, s, a):
        sa = torch.cat([s, a], dim=1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = self.l3(q1)
        return q1


# ----------------- Helper: Fisher–Rao distance (diag Gaussian) -----------------
def fisher_rao_diag(mean_a, logstd_a, mean_b, logstd_b, eps=1e-6):
    """
    FR distance for product of 1D Gaussians (diagonal), exact closed-form.
    Returns batch-mean of sum over action dims of d_i^2.
    d_i = sqrt(2) * acosh( 1 + ((Δμ)^2 + 2(Δσ)^2) / (4 σ_a σ_b) )
    """
    std_a = torch.exp(logstd_a)
    std_b = torch.exp(logstd_b)

    dmu2 = (mean_a - mean_b).pow(2)
    dsig2 = (std_a - std_b).pow(2)
    denom = 4.0 * std_a * std_b + eps

    arg = 1.0 + (dmu2 + 2.0 * dsig2) / denom
    arg = torch.clamp(arg, min=1.0 + eps)  # acosh domain safety

    di = torch.sqrt(torch.tensor(2.0, device=mean_a.device)) * torch.acosh(arg)
    fr2 = di.pow(2).sum(dim=-1).mean()
    return fr2

# ---- kernel utils (RBF MMD in u-space) ---------------------------------
def _pairwise_sq_dists(x, y):
    # x: [N, D], y: [M, D]
    x2 = (x**2).sum(dim=1, keepdim=True)   # [N,1]
    y2 = (y**2).sum(dim=1, keepdim=True).t()  # [1,M]
    return x2 + y2 - 2.0 * (x @ y.t())

def rbf_kernel(x, y, sigma2):
    d2 = _pairwise_sq_dists(x, y).clamp_min(0.0)
    return torch.exp(-d2 / (2.0 * sigma2))

def mmd2_rbf(x, y, sigma2):
    # Unbiased estimator with within/between terms
    n, m = x.size(0), y.size(0)
    Kxx = rbf_kernel(x, x, sigma2)
    Kyy = rbf_kernel(y, y, sigma2)
    Kxy = rbf_kernel(x, y, sigma2)
    # remove diagonals for unbiased estimate
    if n > 1:
        Kxx = (Kxx.sum() - Kxx.diag().sum()) / (n*(n-1))
    else:
        Kxx = torch.tensor(0.0, device=x.device)
    if m > 1:
        Kyy = (Kyy.sum() - Kyy.diag().sum()) / (m*(m-1))
    else:
        Kyy = torch.tensor(0.0, device=y.device)
    Kxy = Kxy.mean()
    return Kxx + Kyy - 2.0 * Kxy

def atanh_clip(x, eps=EPS):
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


# ----------------- TD3-style Gradient-Flow + FR regularizer -----------------
class TD3_GFlow_FR_Agent:
    """
    TD3+BC (gradient-flow view) with Fisher–Rao penalty:
      - Behavior Gaussian regression (MLE)
      - Actor Gaussian + FR(π, π_b) penalty  (replaces W2)
      - Extra entropy reward on actor logits (as before)
      - Critic/target update = TD3
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
        fr_weight=1.0,
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
        self.fr_weight = fr_weight
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

        # ----- Critic (TD3 style) -----
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

        # ----- Behavior (1-step MLE) -----
        behavior_loss = self._train_behavior_policy_one_step(replay_buffer, batch_size)

        metrics = {"critic_loss": float(critic_loss.item()), "behavior_loss": behavior_loss}

        # ----- Actor (delayed) with Fisher–Rao penalty -----
        if self.total_it % self.policy_freq == 0:
            mean_a, log_std_a = self.actor.forward(state)
            Q = self.critic.Q1(state, mean_a)
            lmbda = self.alpha / Q.abs().mean().detach()

            with torch.no_grad():
                mean_b, log_std_b = self.behavior_policy.forward(state)

            # Fisher–Rao (diag Gaussian product) penalty
            fr2 = fisher_rao_diag(mean_a, log_std_a, mean_b, log_std_b)

            # Gaussian entropy of actor (analytical, unsquashed)
            entropy = 0.5 * (2 * log_std_a + np.log(2 * np.pi * np.e)).sum(dim=1).mean()

            actor_loss = self.fr_weight * fr2 - lmbda * Q.mean() - self.entropy_weight * entropy

            self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()

            # soft-update targets
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            metrics.update({
                "actor_loss": float(actor_loss.item()),
                "lambda": float(lmbda.item()),
                "Q_mean": float(Q.mean().item()),
                "fisher_rao_sq": float(fr2.item()),
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
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=device))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=device))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=device))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=device))
        self.actor_target = copy.deepcopy(self.actor)
        self.behavior_policy.load_state_dict(torch.load(filename + "_behavior", map_location=device))
        self.behavior_optimizer.load_state_dict(torch.load(filename + "_behavior_optimizer", map_location=device))

# ----------------- TD3-style Gradient-Flow + FR regularizer -----------------
class TD3_GFlow_FR_Advanced_Agent:
    """
    TD3+BC (gradient-flow view) with Fisher–Rao penalty:
      - Behavior Gaussian regression (MLE)
      - Actor Gaussian + FR(π, π_b) penalty  (replaces W2)
      - Extra entropy reward on actor logits (as before)
      - Critic/target update = TD3
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
        fr_weight=1.0,
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
        self.fr_weight = fr_weight
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

        # ----- Critic (TD3 style) -----
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

        # ----- Behavior (1-step MLE) -----
        behavior_loss = self._train_behavior_policy_one_step(replay_buffer, batch_size)

        metrics = {"critic_loss": float(critic_loss.item()), "behavior_loss": behavior_loss}

        # ----- Actor (delayed) with Fisher–Rao penalty -----
        if self.total_it % self.policy_freq == 0:
            mean_a, log_std_a = self.actor.forward(state)
            Q = self.critic.Q1(state, mean_a)
            lmbda = self.alpha / Q.abs().mean().detach()

            with torch.no_grad():
                mean_b, log_std_b = self.behavior_policy.forward(state)

            # Fisher–Rao (diag Gaussian product) penalty
            fr2 = fisher_rao_diag(mean_a, log_std_a, mean_b, log_std_b)

            # Gaussian entropy of actor (analytical, unsquashed)
            entropy = 0.5 * (2 * log_std_a + np.log(2 * np.pi * np.e)).sum(dim=1).mean()

            actor_loss = self.fr_weight * fr2 - lmbda * Q.mean() - self.entropy_weight * entropy

            self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()

            # soft-update targets
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            metrics.update({
                "actor_loss": float(actor_loss.item()),
                "lambda": float(lmbda.item()),
                "Q_mean": float(Q.mean().item()),
                "fisher_rao_sq": float(fr2.item()),
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
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=device))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=device))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=device))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=device))
        self.actor_target = copy.deepcopy(self.actor)
        self.behavior_policy.load_state_dict(torch.load(filename + "_behavior", map_location=device))
        self.behavior_optimizer.load_state_dict(torch.load(filename + "_behavior_optimizer", map_location=device))

class TD3_GFlow_BC_FRKernel_Agent:
    """
    Kernel-approx FR penalty (MMD in u-space) with your module interfaces.
      - Actor/Behavior: 너가 준 ActorGaussian/BehaviorPolicy 그대로 사용
      - Actor 손실: mmd_weight * MMD^2_u(π_actor, π_beh) - λ * Q - entropy_weight * H
      - u-space 변환: a -> u = atanh(a / max_action)
      - Critic/Target: TD3 스타일
    """
    def __init__(
        self,
        state_dim, action_dim, max_action,
        discount=0.99, tau=0.005,
        policy_noise=0.2, noise_clip=0.5, policy_freq=2,
        alpha=2.5,                 # TD3+BC의 λ 스케일
        mmd_weight=1.0,            # kernel-approx FR 강도
        entropy_weight=0.01,       # 가우시안 엔트로피 보상(unsquashed head)
        mmd_sigma2=1.0,            # RBF 커널 폭^2 (u-space)
        mmd_samples=1,             # 정책당 샘플 수 (1~4 권장)
        grad_clip_norm=None,       # 선택적 안정화
    ):

        self.actor = ActorGaussian(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.behavior_policy = BehaviorPolicy(state_dim, action_dim).to(device)
        self.behavior_optimizer = torch.optim.Adam(self.behavior_policy.parameters(), lr=3e-4)

        self.max_action = float(max_action)
        self.discount, self.tau = discount, tau
        self.policy_noise, self.noise_clip = policy_noise, noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.mmd_weight = mmd_weight
        self.entropy_weight = entropy_weight
        self.mmd_sigma2 = mmd_sigma2
        self.mmd_samples = int(mmd_samples)
        self.grad_clip_norm = grad_clip_norm
        self.total_it = 0

    # ---- utils ----------------------------------------------------------
    def _sample_actions(self, mean, log_std, n_samples):
        """Gaussian head에서 action 샘플. Actor는 mean이 이미 tanh*max라 범위 내/근처. Behavior는 범위를 벗어날 수 있어 clip."""
        if n_samples <= 0:
            return mean
        std = torch.exp(log_std)
        # [B,D] -> [B*n, D]
        eps = torch.randn(mean.size(0), n_samples, mean.size(1), device=mean.device)
        a = mean.unsqueeze(1) + std.unsqueeze(1) * eps
        a = a.reshape(-1, mean.size(1))
        a = torch.clamp(a, -self.max_action + 1e-6, self.max_action - 1e-6)
        return a

    def _to_u_space(self, a):
        """a in [-max,max] -> u in R via atanh(a/max)."""
        return atanh_clip(a / self.max_action)

    # ---- API -------------------------------------------------------------
    def select_action(self, state):
        s = torch.FloatTensor(state.reshape(1, -1)).to(device)
        mean, _ = self.actor.forward(s)
        return mean.detach().cpu().numpy().flatten()

    def _train_behavior_one_step(self, replay_buffer, batch_size):
        s, a, _, _, _ = replay_buffer.sample(batch_size)
        loss = self.behavior_policy.nll(s, a)
        self.behavior_optimizer.zero_grad()
        loss.backward()
        self.behavior_optimizer.step()
        return float(loss.item())

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # ----- Critic (TD3) -----
        s, a, s2, r, not_done = replay_buffer.sample(batch_size)
        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            m2, _ = self.actor_target.forward(s2)
            a2 = (m2 + noise).clamp(-self.max_action, self.max_action)
            q1_t, q2_t = self.critic_target(s2, a2)
            target_Q = r + not_done * self.discount * torch.min(q1_t, q2_t)

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()

        # ----- Behavior (MLE) -----
        behavior_loss = self._train_behavior_one_step(replay_buffer, batch_size)

        metrics = {
            "critic_loss": float(critic_loss.item()),
            "behavior_loss": behavior_loss
        }

        # ----- Actor (delayed): kernel-approx FR via MMD in u-space -----
        if self.total_it % self.policy_freq == 0:
            mean_a, logstd_a = self.actor.forward(s)
            # 평가용 Q는 deterministic mean으로
            Q = self.critic.Q1(s, mean_a)
            lmbda = self.alpha / Q.abs().mean().detach()

            with torch.no_grad():
                mean_b, logstd_b = self.behavior_policy.forward(s)

            # 정책에서 action 샘플 후 u-space 변환
            a_samp = self._sample_actions(mean_a, logstd_a, self.mmd_samples)   # [B*n, D]
            b_samp = self._sample_actions(mean_b, logstd_b, self.mmd_samples)   # [B*n, D]
            u_a = self._to_u_space(a_samp)   # [B*n, D]
            u_b = self._to_u_space(b_samp)   # [B*n, D]

            # RBF-MMD^2 in u-space (kernel-approx FR)
            mmd2_u = mmd2_rbf(u_a, u_b, sigma2=self.mmd_sigma2)

            # Gaussian entropy(unsquashed head) — 네 인터페이스 그대로
            entropy = 0.5 * (2 * logstd_a + np.log(2*np.pi*np.e)).sum(dim=1).mean()

            actor_loss = self.mmd_weight * mmd2_u - lmbda * Q.mean() - self.entropy_weight * entropy

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
            self.actor_optimizer.step()

            # soft-update targets
            with torch.no_grad():
                for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                    tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
                for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                    tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            metrics.update({
                "actor_loss": float(actor_loss.item()),
                "lambda": float(lmbda.item()),
                "Q_mean": float(Q.mean().item()),
                "mmd2_u": float(mmd2_u.item()),
                "entropy": float(entropy.item()),
            })

        return metrics

    # ---- IO --------------------------------------------------------------
    def save(self, prefix):
        torch.save(self.critic.state_dict(),           prefix + "_critic")
        torch.save(self.critic_optimizer.state_dict(), prefix + "_critic_optim")
        torch.save(self.actor.state_dict(),            prefix + "_actor")
        torch.save(self.actor_optimizer.state_dict(),  prefix + "_actor_optim")
        torch.save(self.behavior_policy.state_dict(),  prefix + "_behavior")
        torch.save(self.behavior_optimizer.state_dict(), prefix + "_behavior_optim")
        torch.save({
            "discount": self.discount, "tau": self.tau,
            "policy_noise": self.policy_noise, "noise_clip": self.noise_clip, "policy_freq": self.policy_freq,
            "alpha": self.alpha, "mmd_weight": self.mmd_weight, "entropy_weight": self.entropy_weight,
            "mmd_sigma2": self.mmd_sigma2, "mmd_samples": self.mmd_samples,
            "grad_clip_norm": self.grad_clip_norm, "max_action": self.max_action,
        }, prefix + "_frkernel_misc")

    def load(self, prefix):
        self.critic.load_state_dict(torch.load(prefix + "_critic", map_location=device))
        self.critic_optimizer.load_state_dict(torch.load(prefix + "_critic_optim", map_location=device))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(prefix + "_actor", map_location=device))
        self.actor_optimizer.load_state_dict(torch.load(prefix + "_actor_optim", map_location=device))
        self.actor_target = copy.deepcopy(self.actor)

        self.behavior_policy.load_state_dict(torch.load(prefix + "_behavior", map_location=device))
        self.behavior_optimizer.load_state_dict(torch.load(prefix + "_behavior_optim", map_location=device))

        misc = torch.load(prefix + "_frkernel_misc", map_location=device)
        self.discount      = misc.get("discount", self.discount)
        self.tau           = misc.get("tau", self.tau)
        self.policy_noise  = misc.get("policy_noise", self.policy_noise)
        self.noise_clip    = misc.get("noise_clip", self.noise_clip)
        self.policy_freq   = misc.get("policy_freq", self.policy_freq)
        self.alpha         = misc.get("alpha", self.alpha)
        self.mmd_weight    = misc.get("mmd_weight", self.mmd_weight)
        self.entropy_weight= misc.get("entropy_weight", self.entropy_weight)
        self.mmd_sigma2    = misc.get("mmd_sigma2", self.mmd_sigma2)
        self.mmd_samples   = misc.get("mmd_samples", self.mmd_samples)
        self.grad_clip_norm= misc.get("grad_clip_norm", self.grad_clip_norm)
        self.max_action    = misc.get("max_action", self.max_action)