import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Fisher–Rao utilities (exact geodesic for diagonal Gaussians)
# ============================================================
def _safe_log(x, min_val: float = 1e-12):
    return torch.log(torch.clamp(x, min=min_val))

def hellinger_affinity_diag_gaussians(mean1, logstd1, mean2, logstd2, eps: float = 1e-8):
    """
    Affinity = ∫ sqrt(N1 * N2) dx  for multivariate diagonal Gaussians.
    Σ1 = diag(σ1^2), Σ2 = diag(σ2^2), Σ = 0.5(Σ1+Σ2)
    BC = (|Σ1|^{1/4} |Σ2|^{1/4}) / |Σ|^{1/2} * exp( -1/8 (μ1-μ2)^T Σ^{-1} (μ1-μ2) )
    Inputs: mean*, logstd*: (B,A)
    Returns: (B,) in (0,1]
    """
    var1 = torch.exp(2.0 * logstd1) + eps
    var2 = torch.exp(2.0 * logstd2) + eps
    Sigma = 0.5 * (var1 + var2)                    # (B,A)
    inv_Sigma = 1.0 / Sigma

    dmu = mean1 - mean2                            # (B,A)
    quad = (inv_Sigma * (dmu ** 2)).sum(dim=1)     # (B,)

    # log-determinants (diagonal -> sum of logs)
    logdet_Sigma  = _safe_log(Sigma).sum(dim=1)
    logdet_S1     = (2.0 * logstd1).sum(dim=1)
    logdet_S2     = (2.0 * logstd2).sum(dim=1)

    # log BC = -1/8 * quad - 1/2 log|Σ| + 1/4 (log|Σ1| + log|Σ2|)
    log_BC = -0.125 * quad - 0.5 * logdet_Sigma + 0.25 * (logdet_S1 + logdet_S2)
    log_BC = torch.clamp(log_BC, min=-60.0, max=0.0)  # numerical guard
    return torch.exp(log_BC)

def fisher_rao_distance_diag_gaussians(mean1, logstd1, mean2, logstd2) -> torch.Tensor:
    """
    Exact Fisher–Rao geodesic distance for diagonal Gaussians (B,).
    d_FR = 2 * arccos( ∫ sqrt(p1 p2) dx ).
    """
    bc = hellinger_affinity_diag_gaussians(mean1, logstd1, mean2, logstd2)
    bc = torch.clamp(bc, min=1e-12, max=1.0)  # arccos domain guard
    return 2.0 * torch.acos(bc)

def fisher_rao_squared_diag_gaussians(mean1, logstd1, mean2, logstd2) -> torch.Tensor:
    """Squared FR distance, useful for JKO-style proximal term."""
    d = fisher_rao_distance_diag_gaussians(mean1, logstd1, mean2, logstd2)
    return d ** 2


class BehaviorPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)           # ReBRAC: three layers
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
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
    # Three Layers
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)           # ReBRAC: three layers
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))                  # ReBRAC: three layers
        mean = self.max_action * torch.tanh(self.mean_head(x))
        log_std = self.log_std_head(x).clamp(-5, 2)
        return mean, log_std


class Critic(nn.Module):
    # Three Layers + LayerNorm
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.n1 = nn.LayerNorm(256)
        self.l2 = nn.Linear(256, 256)
        self.n2 = nn.LayerNorm(256)
        self.l3 = nn.Linear(256, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.n4 = nn.LayerNorm(256)
        self.l5 = nn.Linear(256, 256)
        self.n5 = nn.LayerNorm(256)
        self.l6 = nn.Linear(256, 1)

        self.l1b = nn.Linear(256, 256)
        self.n1b = nn.LayerNorm(256)
        self.l4b = nn.Linear(256, 256)
        self.n4b = nn.LayerNorm(256)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.n1(self.l1(sa)))
        q1 = F.relu(self.n1b(self.l1b(q1)))
        q1 = F.relu(self.n2(self.l2(q1)))
        q1 = self.l3(q1)

        q2 = F.relu(self.n4(self.l4(sa)))
        q2 = F.relu(self.n4b(self.l4b(q2)))
        q2 = F.relu(self.n5(self.l5(q2)))
        q2 = self.l6(q2)
        
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.n1(self.l1(sa)))
        q1 = F.relu(self.n1b(self.l1b(q1)))
        q1 = F.relu(self.n2(self.l2(q1)))
        q1 = self.l3(q1)

        return q1


class POGO_FR:
    """
    ReBRAC 베이스:
      - Actor BC(β1)와 Critic 타깃 페널티(β2) 분리
      - Critic 3층 + LayerNorm
      - (옵션) 기존 W2 penalty와 entropy 보상도 함께 사용 가능
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
        entropy_weight=0.01,
        beta_target=0.01,
        # Fisher–Rao options
        fr_weight=1.2,               # ≈ 1/(2τ) 역할
        use_natgrad=True,
        learning_rate=0.001,
    ):
        self.actor = ActorGaussian(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.behavior_policy = BehaviorPolicy(state_dim, action_dim).to(device)
        self.behavior_optimizer = torch.optim.Adam(self.behavior_policy.parameters(), lr=learning_rate)

        # TD3 hyperparams
        self.max_action = max_action
        self.discount   = discount
        self.tau        = tau
        self.policy_noise = policy_noise
        self.noise_clip   = noise_clip
        self.policy_freq  = policy_freq
        self.beta_target  = beta_target
        self.entropy_weight = entropy_weight

        # FR options
        self.fr_weight  = fr_weight
        self.use_natgrad = use_natgrad

        self._actor_prev = copy.deepcopy(self.actor)
        self.total_it = 0

    @torch.no_grad()
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

            # ReBRAC: 타깃 내부에 off-support 억제항 추가 (β2 * ||a' - â'(s')||^2)
            a_hat_prime, _ = self.behavior_policy.forward(next_state)  # mean만 사용
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            penalty = ((next_action - a_hat_prime)**2).sum(dim=1, keepdim=True)  # (B,1)
            
            target_Q = reward + not_done * self.discount * (torch.min(target_Q1, target_Q2) - self.beta_target * penalty)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad(); critic_loss.backward(); self.critic_optimizer.step()

        # ----- Behavior -----
        behavior_loss = self._train_behavior_policy_one_step(replay_buffer, batch_size)

        metrics = {"critic_loss": float(critic_loss.item()), "behavior_loss": behavior_loss}

        # ----- Actor (delayed) -----
        if self.total_it % self.policy_freq == 0:
            mean_a, log_std_a = self.actor.forward(state)
            Q = self.critic.Q1(state, mean_a)
            lmbda = 1.0 / Q.abs().mean().detach()

            with torch.no_grad():
                mean_b, log_std_b = self.behavior_policy.forward(state)
            fr_sq = fisher_rao_squared_diag_gaussians(mean_a, log_std_a, mean_b, log_std_b)  # (B,)
            prox_term = fr_sq.mean()

            entropy = 0.5 * (2 * log_std_a + np.log(2 * np.pi * np.e)).sum(dim=1).mean()

            actor_loss = self.fr_weight * prox_term - lmbda * Q.mean() - self.entropy_weight * entropy

            # ===== Natural Gradient (output hooks) =====
            if self.use_natgrad:
                sigma2 = torch.exp(2.0 * log_std_a).detach().clamp(1e-6, 1e2)  # (B,A)
                h_mu = mean_a.register_hook(lambda g: g * sigma2)  # μ-block: F^{-1}=diag(σ^2)
                h_ls = log_std_a.register_hook(lambda g: g * 0.5)  # logσ-block: F^{-1}=0.5 I
            else:
                h_mu = h_ls = None

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if h_mu is not None: h_mu.remove()
            if h_ls is not None: h_ls.remove()
            self.actor_optimizer.step()

            # # --- Natural Gradient preconditioning (FR local metric) ---
            # if self.use_natgrad:
            #     with torch.no_grad():
            #         # (B, A) 분산 = exp(2*log_std)
            #         sigma2_batch = torch.exp(2.0 * log_std_a)  # shape: (B, A)
            #         sigma2 = sigma2_batch.mean(dim=0)          # shape: (A,)
            #         # sigma2 = sigma2_batch.median(dim=0).values  # 아웃라이어에 민감하면 이걸로

            #         for name, p in self.actor.named_parameters():
            #             if p.grad is None:
            #                 continue

            #             if "mean_head.weight" in name:
            #                 # p.grad: (A, hidden)
            #                 p.grad.mul_(sigma2.view(-1, 1))

            #             elif "mean_head.bias" in name:
            #                 # p.grad: (A,)
            #                 p.grad.mul_(sigma2)

            #             else:
            #                 # 나머지 레이어는 일반 그라디언트 사용
            #                 pass

            # self.actor_optimizer.step()

            # target soft-updates
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            metrics.update({
                "actor_loss": float(actor_loss.item()),
                "lambda": float(lmbda.item()),
                "Q_mean": float(Q.mean().item()),
                "fr_distance2": float(prox_term.item()),
                "entropy": float(entropy.item()),
                "use_natgrad": self.use_natgrad,
            })

        return metrics

    # ---------- I/O ----------
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
