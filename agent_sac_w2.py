import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
EPS = 1e-6


# ----------------- Nets -----------------
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
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def nll(self, state, action):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        var = std**2 + 1e-6
        nll = 0.5 * (((action - mean) ** 2) / var + 2 * log_std + np.log(2 * np.pi))
        return nll.sum(dim=1).mean()


class ActorGaussian(nn.Module):
    """Tanh-squashed diagonal Gaussian with log_prob."""
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
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    @staticmethod
    def _tanh_squash(u):
        a = torch.tanh(u)
        log_det_jac = torch.log(1.0 - a.pow(2) + EPS)
        return a, log_det_jac

    def sample_and_logprob(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        noise = torch.randn_like(mean)
        u = mean + std * noise
        a, log_det = self._tanh_squash(u)
        a = a * self.max_action
        base_logp = (-0.5 * (((u - mean) / (std + EPS)) ** 2 + 2 * log_std + np.log(2*np.pi))).sum(-1)
        logp = base_logp - log_det.sum(-1)
        return a, logp, mean, log_std

    def deterministic(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean) * self.max_action


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


# ----------------- SAC + W2 Agent -----------------
class SAC_W2_Agent:
    """
    SAC 스타일 정책 + W2 정규화(behavior policy와의 거리):
      - Policy loss:  E[ alpha*logpi - Q_min ] + w2_weight * W2^2
      - Target: r + γ(1-d) * (min Q_tgt(s',a') - alpha*logpi(a'|s'))
      - Alpha 자동 튜닝(기본 target_entropy = -action_dim)
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_freq=1,
        w2_weight=1.0,
        alpha_init=0.2,
        autotune_alpha=True,
        target_entropy=None
    ):
        self.actor = ActorGaussian(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.behavior = BehaviorPolicy(state_dim, action_dim).to(device)
        self.behavior_opt = torch.optim.Adam(self.behavior.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq
        self.w2_weight = w2_weight

        self.target_entropy = (-float(action_dim)) if target_entropy is None else target_entropy
        self.log_alpha = torch.tensor(np.log(alpha_init), requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.autotune_alpha = autotune_alpha

        self.total_it = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if deterministic:
            a = self.actor.deterministic(state)
        else:
            a, _, _, _ = self.actor.sample_and_logprob(state)
        return a.cpu().data.numpy().flatten()

    def _train_behavior_step(self, replay_buffer, batch_size):
        s, a, _, _, _ = replay_buffer.sample(batch_size)
        loss = self.behavior.nll(s, a)
        self.behavior_opt.zero_grad()
        loss.backward()
        self.behavior_opt.step()
        return float(loss.item())

    @staticmethod
    def _w2_diag(mean_a, logstd_a, mean_b, logstd_b):
        std_a = torch.exp(logstd_a)
        std_b = torch.exp(logstd_b)
        term_mean = (mean_a - mean_b).pow(2).sum(-1)
        term_std  = (std_a - std_b).pow(2).sum(-1)
        return (term_mean + term_std).mean()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        s, a, s2, r, not_done = replay_buffer.sample(batch_size)

        # ----- Critic (SAC target) -----
        with torch.no_grad():
            a2, logp2, _, _ = self.actor.sample_and_logprob(s2)
            tq1, tq2 = self.critic_target(s2, a2)
            v2 = torch.min(tq1, tq2) - self.alpha.detach() * logp2
            y = r + not_done * self.discount * v2

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        # add cql terms
        with torch.no_grad():
            a_rand = torch.empty_like(a).uniform_(-self.max_action, self.max_action)
            a_pi, _, _, _ = self.actor.sample_and_logprob(s)
        # Q on sampled
        q1_rand, q2_rand = self.critic(s, a_rand)
        q1_pi,   q2_pi   = self.critic(s, a_pi)
        # logsumexp over K=2 samples (rand, pi) -- 더 넣어도 됨
        q1_cat = torch.stack([q1_rand, q1_pi], dim=0)  # [K,B,1]
        q2_cat = torch.stack([q2_rand, q2_pi], dim=0)
        cql_term = (torch.logsumexp(q1_cat, dim=0) - q1 + torch.logsumexp(q2_cat, dim=0) - q2).mean()
        critic_loss = critic_loss + 0.5 * cql_term #나중에 하이퍼파라미터화

        self.critic_optimizer.zero_grad(); critic_loss.backward(); self.critic_optimizer.step()

        # ----- Behavior MLE -----
        behavior_loss = self._train_behavior_step(replay_buffer, batch_size)

        metrics = {
            "critic_loss": float(critic_loss.item()),
            "behavior_loss": behavior_loss,
            "alpha": float(self.alpha.item()),
        }

        # ----- Actor + alpha -----
        if self.total_it % self.policy_freq == 0:
            new_a, logp, mean_a, logstd_a = self.actor.sample_and_logprob(s)
            q1_pi, q2_pi = self.critic(s, new_a)
            q_pi = torch.min(q1_pi, q2_pi)

            with torch.no_grad():
                mean_b, logstd_b = self.behavior(s)
            w2 = self._w2_diag(mean_a, logstd_a, mean_b, logstd_b)

            policy_loss = (self.alpha.detach() * logp - q_pi).mean() + self.w2_weight * w2
            self.actor_optimizer.zero_grad(); policy_loss.backward(); self.actor_optimizer.step()

            metrics.update({
                "actor_loss": float(policy_loss.item()),
                "policy_logp": float(logp.mean().item()),
                "Q_pi_mean": float(q_pi.mean().item()),
                "w2_distance": float(w2.item()),
            })

            if self.autotune_alpha:
                alpha_loss = (self.alpha * (-logp.detach() - self.target_entropy)).mean()
                self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
                metrics["alpha_loss"] = float(alpha_loss.item())

            # soft update
            with torch.no_grad():
                for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                    tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
                for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                    tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return metrics

    # ----- IO -----
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_opt")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_opt")
        torch.save(self.behavior.state_dict(), filename + "_behavior")
        torch.save(self.behavior_opt.state_dict(), filename + "_behavior_opt")
        torch.save({
            "log_alpha": float(self.log_alpha.detach().cpu().item()),
            "w2_weight": self.w2_weight,
            "tau": self.tau,
            "discount": self.discount,
            "target_entropy": self.target_entropy,
        }, filename + "_misc")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=device))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_opt", map_location=device))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=device))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_opt", map_location=device))
        self.actor_target = copy.deepcopy(self.actor)
        self.behavior.load_state_dict(torch.load(filename + "_behavior", map_location=device))
        self.behavior_opt.load_state_dict(torch.load(filename + "_behavior_opt", map_location=device))
        misc = torch.load(filename + "_misc", map_location=device)
        self.log_alpha = torch.tensor(misc.get("log_alpha", np.log(0.2)), requires_grad=True, device=device)
        self.w2_weight = misc.get("w2_weight", self.w2_weight)
        self.tau = misc.get("tau", self.tau)
        self.discount = misc.get("discount", self.discount)
        self.target_entropy = misc.get("target_entropy", self.target_entropy)
