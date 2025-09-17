# main_two.py
# 2-step gradient flow training:
# Use the previously trained (1-step) actor as a frozen reference distribution
# by replacing behavior_policy with a FrozenActorRef, then continue training.

import os
import argparse
import numpy as np
import torch
import gym
import d4rl

import utils
import agent  # for TD3_BC baseline (optional)

try:
    from agent_gflow import TD3_GFlow_BC, ActorGaussian
except Exception:
    TD3_GFlow_BC = None
    ActorGaussian = None

# -----------------------------
# optional wandb
# -----------------------------
try:
    import wandb
except ImportError:
    wandb = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    env = gym.make(env_name)
    env.seed(seed + seed_offset)
    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, done = env.reset(), False
        while not done:
            s = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(s)
            state, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = env.get_normalized_score(avg_reward) * 100.0
    return float(avg_reward), float(d4rl_score)


class FrozenActorRef(torch.nn.Module):
    """
    Frozen reference policy: wraps a frozen copy of an ActorGaussian
    and exposes forward(state) -> (mean, logstd) just like behavior_policy.
    """
    def __init__(self, actor_gaussian: ActorGaussian):
        super().__init__()
        import copy
        self.ref = copy.deepcopy(actor_gaussian).to(device).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, state):
        # ActorGaussian.forward returns (mean, logstd)
        return self.ref(state)


def main():
    parser = argparse.ArgumentParser(description="Two-step gradient flow runner")
    # basic
    parser.add_argument("--policy", default="TD3_GFlow_BC", choices=["TD3_GFlow_BC", "TD3_BC"],
                        help="Two-step은 주로 TD3_GFlow_BC에 해당")
    parser.add_argument("--env", default="hopper-medium-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_timesteps", type=int, default=1000000, help="2nd step training timesteps")
    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument("--normalize", type=str, default="True")

    # TD3 core
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_freq", type=int, default=2)

    # TD3+BC / GFlow
    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--w2_weight", type=float, default=1.0)
    parser.add_argument("--entropy_weight", type=float, default=0.01)

    # IO
    parser.add_argument("--models_dir", default="./models")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--load_model", required=True,
                        help='Base filename of the 1-step model to load, e.g., TD3_GFlow_BC_hopper-medium-v2_0')
    parser.add_argument("--save_suffix", default="_2step",
                        help='Suffix for saving the 2-step model')

    # wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="td3-suite")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--run_name", default="")

    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # env & seeds
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = dict(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=args.discount,
        tau=args.tau,
        policy_noise=args.policy_noise * max_action,
        noise_clip=args.noise_clip * max_action,
        policy_freq=args.policy_freq,
        alpha=args.alpha,
    )

    # init policy and load step-1 model
    if args.policy == "TD3_GFlow_BC":
        assert TD3_GFlow_BC is not None, "agent_gflow.py is required."
        policy = TD3_GFlow_BC(**kwargs)
    else:
        # TD3_BC fallback: 2-step 의미는 약하지만 로드 후 추가 학습 가능
        policy = agent.TD3_BC(**kwargs)

    load_base = os.path.join(args.models_dir, args.load_model)
    policy.load(load_base)
    print(f"[load] loaded 1-step model from: {load_base}_*")

    # replay buffer
    rb = utils.ReplayBuffer(state_dim, action_dim)
    rb.convert_D4RL(d4rl.qlearning_dataset(env))
    if args.normalize.lower() in ["true", "1", "yes", "y"]:
        mean, std = rb.normalize_states()
    else:
        mean, std = 0, 1

    # ====== Two-step trick: replace behavior with frozen previous-actor ======
    if args.policy == "TD3_GFlow_BC":
        # Wrap the CURRENT (loaded) actor as reference, freeze it, and plug it in.
        # Now W2 compares (current actor) vs (frozen previous actor).
        frozen_ref = FrozenActorRef(policy.actor)
        policy.behavior_policy = frozen_ref  # monkey-patch: W2 term will use this instead of dataset behavior

        # Also disable behavior fitting to avoid updating the frozen reference
        # (agent_gflow.train() calls _train_behavior_policy_one_step every iteration)
        def _noop_behavior_train(replay_buffer, batch_size):
            return 0.0
        policy._train_behavior_policy_one_step = _noop_behavior_train
        print("[two-step] behavior_policy replaced with frozen previous-actor; behavior training disabled.")

    # ====== wandb ======
    run = None
    if args.wandb and wandb is not None:
        run_name = args.run_name if args.run_name else f"{args.env}_{args.policy}_2step_seed{args.seed}"
        init_kwargs = dict(project=args.wandb_project, name=run_name, group=args.env,
                           tags=[args.policy, "2step"], config=vars(args))
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        try:
            run = wandb.init(**init_kwargs)
        except Exception as e:
            print(f"[wandb] init failed: {e}")
            run = None

    # ====== training loop (2nd step) ======
    total_steps = int(args.max_timesteps)
    evaluations = []

    # pre-eval
    pre_avg, pre_d4rl = eval_policy(policy, args.env, args.seed, mean, std)
    print(f"[2step] Before: avg_reward={pre_avg:.2f}, d4rl={pre_d4rl:.2f}")
    if run is not None:
        wandb.log({"two_step/before_avg_reward": pre_avg,
                   "two_step/before_d4rl": pre_d4rl,
                   "two_step/d4rl": pre_d4rl, 
                   "eval/d4rl": pre_d4rl, 
                   "timesteps": 0}, step=0)

    for t in range(total_steps):
        metrics = policy.train(rb, args.batch_size)

        # wandb logs
        if run is not None:
            log_data = {"train2/critic_loss": metrics.get("critic_loss", 0.0),
                        "timesteps": t + 1}
            if "behavior_loss" in metrics:
                # should be 0.0 due to _noop, but log anyway
                log_data.update({
                    "train2/behavior_loss": metrics["behavior_loss"],
                    "train2/w2_distance": metrics.get("w2_distance", 0.0),
                    "train2/entropy": metrics.get("entropy", 0.0)
                })
            if "actor_loss" in metrics:
                log_data.update({
                    "train2/actor_loss": metrics["actor_loss"],
                    "train2/lambda": metrics.get("lambda", 0.0),
                    "train2/Q_mean": metrics.get("Q_mean", 0.0),
                })
            wandb.log(log_data, step=t + 1)

        # periodic eval
        if (t + 1) % args.eval_freq == 0:
            avg_r, dscore = eval_policy(policy, args.env, args.seed, mean, std)
            evaluations.append(dscore)
            np.save(os.path.join(args.results_dir, f"{args.load_model}{args.save_suffix}"), evaluations)
            print(f"[2step] steps {t+1}/{total_steps} | avg_reward={avg_r:.2f} | d4rl={dscore:.2f}")
            if run is not None:
                wandb.log({"two_step/avg_reward": avg_r,
                           "eval/d4rl": dscore,
                           "timesteps": t + 1}, step=t + 1)

    # save
    save_base = os.path.join(args.models_dir, f"{args.load_model}{args.save_suffix}")
    policy.save(save_base)
    print(f"[2step] saved model to: {save_base}_*")

    if run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
