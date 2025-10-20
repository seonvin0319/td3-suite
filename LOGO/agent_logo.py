import copy, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BehaviorPolicy(nn.Module):
    """Latent Gaussian β(s): outputs (mu_z, log_std_z). NLL is tanh-Gaussian with CoV."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        mu_z = self.mean_head(x)
        log_std_z = self.log_std_head(x).clamp(-5, 2)
        return mu_z, log_std_z

    def nll(self, s, a, max_action, eps=1e-6):
        """-log p(a|s) for tanh-Gaussian via change of variables (diag)."""
        mu_z, log_std_z = self.forward(s)
        # a -> u in (-1,1) -> z = atanh(u)
        u = (a / max_action).clamp(-1 + eps, 1 - eps)
        z = 0.5 * (torch.log1p(u) - torch.log1p(-u))  # atanh(u)
        std = log_std_z.exp()
        # log N(z; mu, std) sum over dims
        log2pi = math.log(2.0 * math.pi)
        log_pz = -0.5 * (((z - mu_z) / std) ** 2 + 2.0 * log_std_z + log2pi)
        log_pz = log_pz.sum(dim=-1)
        # log |det du/dz| = sum log(1 - u^2)
        log_det = torch.log(1.0 - u * u + eps).sum(dim=-1)
        # nll = -(log pz - log_det)
        return (-(log_pz - log_det)).mean()


class ActorGaussian(nn.Module):
    """Latent Gaussian πθ(s): outputs (mu_z, log_std_z)."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        mu_z = self.mean_head(x)
        log_std_z = self.log_std_head(x).clamp(-5, 2)
        return mu_z, log_std_z


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


class LOGO_W2:
    """
    Latent-space Gradient Flow (TD3+BC flavored)
      - Actor/Behavior are Gaussians in z; actions only when needed: a = max * tanh(z)
      - Distance term: closed-form W2^2 between z-Gaussians (diag)
      - Entropy: z-Gaussian entropy (analytic)
      - Natural gradient: scale log-std grads in z
    """
    def __init__(self,
                 state_dim, action_dim, max_action,
                 discount=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2,
                 w_weight=1.0, entropy_weight=0.01, learning_rate=3e-4,
                 use_natgrad=True):
        
        self.actor = ActorGaussian(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.behavior = BehaviorPolicy(state_dim, action_dim).to(device)
        self.behavior_opt = torch.optim.Adam(self.behavior.parameters(), lr=learning_rate)

        self.max_action = float(max_action)
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.w_weight = w_weight
        self.entropy_weight = entropy_weight
        self.use_natgrad = use_natgrad
        self.total_it = 0

    @torch.no_grad()
    def select_action(self, state, stochastic=False):
        s = torch.FloatTensor(state.reshape(1, -1)).to(device)
        mu_z, log_std_z = self.actor(s)
        if stochastic:
            z = mu_z + log_std_z.exp() * torch.randn_like(mu_z)
        else:
            z = mu_z
        a = self.max_action * torch.tanh(z)
        return a.cpu().numpy().flatten()

    # ---------------- Behavior step (BC NLL) ----------------
    def _train_behavior(self, replay_buffer, batch_size):
        s, a, _, _, _ = replay_buffer.sample(batch_size)
        loss = self.behavior.nll(s, a, self.max_action)
        self.behavior_opt.zero_grad()
        loss.backward()
        self.behavior_opt.step()
        return float(loss.item())

    # ---------------- One training iteration ----------------
    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # ----- Critic -----
        s, a, s2, r, not_done = replay_buffer.sample(batch_size)
        with torch.no_grad():
            mu_t, logstd_t = self.actor_target(s2)
            z_t = mu_t
            noise = torch.randn_like(mu_t) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            z_t = z_t + noise
            a_t = self.max_action * torch.tanh(z_t)

            tq1, tq2 = self.critic_target(s2, a_t)
            target_Q = r + not_done * self.discount * torch.min(tq1, tq2)

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        # ----- Behavior (BC) -----
        behavior_loss = self._train_behavior(replay_buffer, batch_size)

        metrics = {"critic_loss": float(critic_loss.item()), "behavior_loss": behavior_loss}

        # ----- Actor (delayed) -----
        if self.total_it % self.policy_freq == 0:
            mu_a, logstd_a = self.actor(s)
            std_a = logstd_a.exp()
            z = mu_a + std_a * torch.randn_like(mu_a)  
            a_squash = self.max_action * torch.tanh(z)
            Q = self.critic.Q1(s, a_squash)
            lamb = 1.0 / (Q.abs().mean().detach() + 1e-8)

            with torch.no_grad():
                mu_b, logstd_b = self.behavior(s)
                std_b = logstd_b.exp()

            # latent W2^2 (diag): mean^2 + (σ diff)^2; mean over batch and dims
            w2 = ((mu_a - mu_b) ** 2).mean() + ((std_a - std_b) ** 2).mean()

            Hz = 0.5 * (2 * logstd_a + np.log(2 * np.pi * np.e)).sum(dim=1).mean()

            actor_loss = self.w_weight * w2 - lamb * Q.mean() - self.entropy_weight * Hz

            self.actor_opt.zero_grad()
            if self.use_natgrad:
                # natural gradient scaling on log-std block in z
                scale = torch.exp(-2.0 * logstd_a.detach()).clamp(1e-3, 1e3)
                hook = logstd_a.register_hook(lambda g: g * scale)
                actor_loss.backward()
                hook.remove()
            else:
                actor_loss.backward()
            self.actor_opt.step()

            # soft updates
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            metrics.update({
                "actor_loss": float(actor_loss.item()),
                "lambda": float(lamb.item()),
                "Q_mean": float(Q.mean().item()),
                "w2_distance": float(w2.item()),
                "entropy_z": float(Hz.item()),
            })

        return metrics

    # ---------------- I/O ----------------
    def save(self, prefix):
        torch.save(self.critic.state_dict(), prefix + "_critic")
        torch.save(self.critic_opt.state_dict(), prefix + "_critic_optimizer")
        torch.save(self.actor.state_dict(), prefix + "_actor")
        torch.save(self.actor_opt.state_dict(), prefix + "_actor_optimizer")
        torch.save(self.behavior.state_dict(), prefix + "_behavior")
        torch.save(self.behavior_opt.state_dict(), prefix + "_behavior_optimizer")

    def load(self, prefix):
        self.critic.load_state_dict(torch.load(prefix + "_critic", map_location=device))
        self.critic_opt.load_state_dict(torch.load(prefix + "_critic_optimizer", map_location=device))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(prefix + "_actor", map_location=device))
        self.actor_opt.load_state_dict(torch.load(prefix + "_actor_optimizer", map_location=device))
        self.actor_target = copy.deepcopy(self.actor)

        self.behavior.load_state_dict(torch.load(prefix + "_behavior", map_location=device))
        self.behavior_opt.load_state_dict(torch.load(prefix + "_behavior_optimizer", map_location=device))
