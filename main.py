# File: td3_suite/td3w/main.py

import numpy as np
import torch
import gym
import argparse
import os
import d4rl

import utils

try:
    import wandb
except ImportError:
    wandb = None


def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            # normalize
            nstate = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(nstate)   # 각 에이전트가 평균/결정론적 액션을 반환
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100.0

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
    print("---------------------------------------")
    return d4rl_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy",
                        default="TD3_BC",
                        choices=["TD3_BC", "GFlow_W2","TD3_GFlow_BC", "SAC_W2", "GFlow_FR", "GFlow_KL"])
    parser.add_argument("--env", default="hopper-medium-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_freq", type=int, default=5_000)
    parser.add_argument("--max_timesteps", type=int, default=1_000_000)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_model", default="")
    parser.add_argument("--normalize", default=True, type=bool)

    # Shared/TD3
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_freq", type=int, default=1)

    # TD3+BC / GFlow(Baseline)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--w2_weight", type=float, default=0.5)

    parser.add_argument("--final_eval_runs", type=int, default=5)
    parser.add_argument("--final_eval_episodes", type=int, default=10)

    # W&B
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="gflow_offRL")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--run_name", default="")
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    os.makedirs("./results", exist_ok=True)
    if args.save_model:
        os.makedirs("./models", exist_ok=True)

    run = None
    if args.wandb:
        run_name = args.run_name if args.run_name else f"{args.env}_{args.policy}_seed{args.seed}"
        init_kwargs = dict(
            project=args.wandb_project,
            name=run_name,
            group=args.env,
            tags=[args.policy],
            config=vars(args),
        )
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        run = wandb.init(**init_kwargs)
        wandb.log({"_boot": 1, "timesteps": 0}, step=0)

    if args.wandb and wandb is not None and wandb.run is not None:
        # wandb.config에 정의된 sweep 파라미터가 있으면 args에 덮어쓰기
        for k, v in dict(wandb.config).items():
            if hasattr(args, k):
                setattr(args, k, v)

    env = gym.make(args.env)

    # Seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 공통 kwargs
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        "alpha": args.alpha,  # TD3-BC/GFlow 계열에서 사용
    }

    # ---------------- Initialize policy ----------------
    if args.policy == "TD3_BC":
        from agent import TD3_BC
        policy = TD3_BC(**kwargs)

    elif args.policy == "GFlow_W2":
        from agent_gflow import GFlow_W2
        policy = GFlow_W2(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            discount=args.discount,
            tau=args.tau,
            policy_freq=args.policy_freq,
            alpha=1.0,                      # alpha=1 고정 스윕도 가능
            w2_weight=args.w2_weight,
            entropy_weight=args.entropy_weight,
        )

    else:
        raise NotImplementedError(f"Unknown policy: {args.policy}")

    # ---------------- Load / Dataset / Normalize ----------------
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean = np.zeros((1, state_dim), dtype=np.float32)
        std = np.ones((1, state_dim), dtype=np.float32)

    # ---------------- Train Loop ----------------
    evaluations = []
    best_eval = -np.inf
    for t in range(int(args.max_timesteps)):

        metrics = policy.train(replay_buffer, args.batch_size)
        # ------------------------------
        if args.wandb:
            log_data = {"timesteps": t + 1}

            for k in ["critic_loss", "actor_loss", "behavior_loss"]:
                if k in metrics:
                    name = "train/behavior_nll" if k == "behavior_loss" else f"train/{k}"
                    log_data[name] = metrics[k]

            # TD3_GFlow_BC
            for k in ["lambda", "Q_mean", "w2_distance", "fr_distance", "entropy"]:
                if k in metrics:
                    log_data[f"train/{k}"] = metrics[k]

            wandb.log(log_data, step=t + 1)

        # Evaluate
        if (t + 1) % args.eval_freq == 0:
            print(f"Time steps: {t+1}")
            d4rl_score = eval_policy(policy, args.env, args.seed, mean, std)
            evaluations.append(d4rl_score)
            np.save(f"./results/{file_name}", evaluations)

            if args.wandb:
                wandb.log({"eval/d4rl": d4rl_score, "timesteps": t + 1}, step=t + 1)
            if args.save_model:
                policy.save(f"./models/{file_name}")
                if d4rl_score > best_eval:
                    best_eval = d4rl_score
                    policy.save(f"./models/{file_name}_best")

    # -------- Final evaluation on the trained weights --------
    print("======== Final Evaluation (trained weights) ========")
    final_scores = []
    for r in range(args.final_eval_runs):
        score = eval_policy(
            policy, args.env, args.seed, mean, std,
            seed_offset=1000 + 100 * r,            # 평가 난수 고정용 오프셋
            eval_episodes=args.final_eval_episodes
        )
        final_scores.append(score)
    final_scores = np.array(final_scores, dtype=np.float32)
    final_mean, final_std = float(final_scores.mean()), float(final_scores.std())
    print(f"[FINAL] mean={final_mean:.3f}, std={final_std:.3f} "
          f"over {args.final_eval_runs}x{args.final_eval_episodes} episodes")
    if wandb and args.wandb:
        wandb.log({
            "final/eval_mean": final_mean,
            "final/eval_std": final_std,
            "final/runs": args.final_eval_runs,
            "final/episodes": args.final_eval_episodes
        })
        wandb.finish()
