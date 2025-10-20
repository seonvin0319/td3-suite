# File: td3_suite/latent/main_logo.py
import os
import numpy as np
import torch
import gym
import argparse
import d4rl
import utils

try:
    import wandb
except ImportError:
    wandb = None


def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    """평가 루프: normalization 후 deterministic action으로 D4RL 점수 계산."""
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            nstate = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(nstate, stochastic=False)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100.0

    print(f"[Eval] Return: {avg_reward:.3f}, D4RL Score: {d4rl_score:.2f}")
    return d4rl_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="hopper-medium-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument("--max_timesteps", type=int, default=1_000_000)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_model", default="")
    parser.add_argument("--normalize", default=True, type=bool)

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_freq", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--w_weight", type=float, default=1.0)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--use_natgrad", default=True, type=bool)

    # W&B
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="logo_latentW2")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--run_name", default="")

    args = parser.parse_args()

    # --- setup ---
    file_name = f"LOGO_W2_{args.env}_seed{args.seed}"
    print(f"\n=== LOGO_W2 Training on {args.env} (seed {args.seed}) ===")

    os.makedirs("./results", exist_ok=True)
    if args.save_model:
        os.makedirs("./models", exist_ok=True)

    if args.wandb:
        run_name = args.run_name or file_name
        init_kwargs = dict(
            project=args.wandb_project,
            name=run_name,
            group=args.env,
            tags=["LOGO_W2"],
            config=vars(args),
        )
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        wandb.init(**init_kwargs)
        wandb.log({"_boot": 1}, step=0)

    # --- env & seed ---
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # --- agent import ---
    from agent_logo import LOGO_W2
    policy = LOGO_W2(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=args.discount,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        w_weight=args.w_weight,
        entropy_weight=args.entropy_weight,
        learning_rate=args.learning_rate,
        use_natgrad=args.use_natgrad,
    )

    # --- dataset ---
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean = np.zeros((1, state_dim), dtype=np.float32)
        std = np.ones((1, state_dim), dtype=np.float32)

    # --- training ---
    evaluations, best_eval = [], -np.inf
    for t in range(int(args.max_timesteps)):
        metrics = policy.train(replay_buffer, args.batch_size)

        # W&B logging
        if args.wandb:
            log_data = {"timesteps": t + 1}
            for k, v in metrics.items():
                log_data[f"train/{k}"] = v
            wandb.log(log_data, step=t + 1)

        # Evaluation
        if (t + 1) % args.eval_freq == 0:
            score = eval_policy(policy, args.env, args.seed, mean, std)
            evaluations.append(score)
            np.save(f"./results/{file_name}", evaluations)
            if args.wandb:
                wandb.log({"eval/d4rl": score, "timesteps": t + 1}, step=t + 1)
            if args.save_model:
                policy.save(f"./models/{file_name}")
                if score > best_eval:
                    best_eval = score
                    policy.save(f"./models/{file_name}_best")

    print("=== Training Complete ===")
    if args.wandb:
        wandb.finish()
