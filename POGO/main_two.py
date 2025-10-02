import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

from agent_gflow import GFlow_W2, GFlow_W2_Refine, device


def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)
    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            nstate = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(nstate)
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
    parser.add_argument("--policy_freq", type=int, default=2)

    # GFlow (Baseline)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--w2_weight", type=float, default=0.5)

    parser.add_argument("--final_eval_runs", type=int, default=5)
    parser.add_argument("--final_eval_episodes", type=int, default=10)

    # Two-step control (하프 스위치)
    parser.add_argument("--two_step", action="store_true")
    parser.add_argument("--split_ratio", type=float, default=0.5)
    parser.add_argument("--freeze_critic_mode", default=True, type=bool)

    # W&B
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="gflow_offRL")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--start_mode", choices=["scratch", "load"], default="scratch",
                   help="scratch: 모든 파트 새로 시작 / load: 일부 파트를 기존 체크포인트에서 로드")
    parser.add_argument("--load_prefix", type=str, default="",
                   help="예: ./POGO/models/POGO_Two-Step_hopper-medium-v2_0  (확장자 없이 prefix)")

    args = parser.parse_args()

    file_name = f"POGO_Two-Step_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Two-Phase POGO (agent switch), Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    os.makedirs("./POGO/results", exist_ok=True)
    if args.save_model:
        os.makedirs("./POGO/models", exist_ok=True)

    MID_DIR = "./models_mid"
    os.makedirs(MID_DIR, exist_ok=True)
    midpoint = args.max_timesteps // 2
    print(f"[CKPT] Midpoint checkpoint will be saved at step {midpoint} into {MID_DIR}")

    run = None
    if args.wandb:
        run_name = args.run_name if args.run_name else f"{args.env}_POGO_Two-Step_seed{args.seed}"
        init_kwargs = dict(
            project=args.wandb_project,
            name=run_name,
            group=args.env,
            tags=["POGO_Two-Step"],
            config=vars(args),
        )
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        run = wandb.init(**init_kwargs)
        wandb.log({"_boot": 1, "timesteps": 0}, step=0)

    if args.wandb and wandb is not None and wandb.run is not None:
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

    # ---- Dataset / Normalize ----
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean = np.zeros((1, state_dim), dtype=np.float32)
        std  = np.ones((1, state_dim), dtype=np.float32)

    # ===== Agent A (Phase-1) =====
    agentA = GFlow_W2(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=args.discount,
        tau=args.tau,
        policy_noise=args.policy_noise * max_action,
        noise_clip=args.noise_clip * max_action,
        policy_freq=args.policy_freq,
        alpha=args.alpha,
        w2_weight=args.w2_weight,
        entropy_weight=args.entropy_weight,
    )

    # Load to agentA if requested
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        agentA.load(f"./POGO/models/{policy_file}")

    # ---- Train schedule ----
    evaluations = []
    best_eval = -np.inf

    if args.two_step:
        raw_split = int(round(args.max_timesteps * args.split_ratio))
        split_step = max(1, min(args.max_timesteps - 1, raw_split))
    else:
        split_step = args.max_timesteps  # 전부 agentA로 돌고 종료

    # ===== Phase-1: 0 ~ split_step-1 (GFlow_W2) =====
    for t in range(split_step):
        metrics = agentA.train(replay_buffer, args.batch_size)

        if args.wandb:
            log_data = {"timesteps": t + 1, "phase": 1}
            for k in ["critic_loss", "actor_loss", "behavior_loss"]:
                if k in metrics:
                    name = "train/behavior_nll" if k == "behavior_loss" else f"train/{k}"
                    log_data[name] = metrics[k]
            for k in ["lambda", "Q_mean", "w2_distance", "entropy", "actor_std", "behavior_std", "actor_mean", "behavior_mean"]:
                if k in metrics:
                    log_data[f"train/{k}"] = metrics[k]
            wandb.log(log_data, step=t + 1)

        if (t + 1) % args.eval_freq == 0:
            print(f"[Phase-1] Time steps: {t+1}")
            d4rl_score = eval_policy(agentA, args.env, args.seed, mean, std)
            evaluations.append(d4rl_score)
            np.save(f"./POGO/results/{file_name}", evaluations)
            if args.wandb:
                wandb.log({"eval/d4rl": d4rl_score, "timesteps": t + 1, "phase": 1}, step=t + 1)
            if args.save_model:
                agentA.save(f"./POGO/models/{file_name}")
                if d4rl_score > best_eval:
                    best_eval = d4rl_score
                    agentA.save(f"./POGO/models/{file_name}_best")

    
    mid_name = f"{file_name}_mid_{t+1}"
    agentA.save(f"{MID_DIR}/{mid_name}")
    print(f"[CKPT] Saved midpoint checkpoint at step {t+1} -> {MID_DIR}/{mid_name}_*")
    if args.wandb:
        wandb.log({"ckpt/mid_saved_step": t + 1}, step=t + 1)

    # ===== Phase-2: split_step ~ max_timesteps-1 (GFlow_W2_Refine) =====
    remaining = args.max_timesteps - split_step
    if remaining > 0:
        # ===== Agent B (Phase-2) =====
        agentB = GFlow_W2_Refine(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            discount=args.discount,
            tau=args.tau,
            policy_noise=args.policy_noise * max_action,
            noise_clip=args.noise_clip * max_action,
            policy_freq=args.policy_freq,
            alpha=args.alpha,
            w2_weight=args.w2_weight,
            entropy_weight=args.entropy_weight,
        )

        # 1) actor 초기값: A.actor → B.actor
        agentB.actor.load_state_dict(agentA.actor.state_dict())
        agentB.actor_target.load_state_dict(agentA.actor_target.state_dict())

        # 2) critic: A.critic → B.critic (target 포함)
        agentB.critic.load_state_dict(agentA.critic.state_dict())
        agentB.critic_target.load_state_dict(agentA.critic_target.state_dict())

        # 3) reference는 Phase-1 actor 그대로 쓰기 (스케일 일치 보장)
        agentB.behavior_policy.load_state_dict(agentA.actor.state_dict())

        # ===== Phase-2 학습 루프 =====
        for t2 in range(remaining):
            global_step = split_step + t2  # 0-based
            metrics = agentB.train(replay_buffer, args.batch_size)

            if args.wandb:
                log_data = {"timesteps": global_step + 1, "phase": 2}
                for k in ["critic_loss", "actor_loss", "behavior_loss"]:
                    if k in metrics:
                        name = "train/behavior_nll" if k == "behavior_loss" else f"train/{k}"
                        log_data[name] = metrics[k]
                for k in ["lambda", "Q_mean", "w2_distance", "entropy", "actor_std", "behavior_std", "actor_mean", "behavior_mean"]:
                    if k in metrics:
                        log_data[f"train/{k}"] = metrics[k]
                wandb.log(log_data, step=global_step + 1)

            if (global_step + 1) % args.eval_freq == 0:
                print(f"[Phase-2] Time steps: {global_step + 1}")
                d4rl_score = eval_policy(agentB, args.env, args.seed, mean, std)
                evaluations.append(d4rl_score)
                np.save(f"./POGO/results/{file_name}", evaluations)
                if args.wandb:
                    wandb.log({"eval/d4rl": d4rl_score, "timesteps": global_step + 1, "phase": 2}, step=global_step + 1)
                if args.save_model:
                    agentB.save(f"./POGO/models/{file_name}")
                    if d4rl_score > best_eval:
                        best_eval = d4rl_score
                        agentB.save(f"./POGO/models/{file_name}_best")

        # 마지막 평가는 AgentB로
        active_policy = agentB
    else:
        # 전부 AgentA로 끝난 경우
        active_policy = agentA

    # -------- Final evaluation on the trained weights --------
    print("======== Final Evaluation (trained weights) ========")
    final_scores = []
    for r in range(args.final_eval_runs):
        score = eval_policy(
            active_policy, args.env, args.seed, mean, std,
            seed_offset=1000 + 100 * r,
            eval_episodes=args.final_eval_episodes
        )
        final_scores.append(score)
    final_scores = np.array(final_scores, dtype=np.float32)
    final_mean, final_std = float(final_scores.mean()), float(final_scores.std())
    print(f"[FINAL] mean={final_mean:.3f}, std={final_std:.3f} over {args.final_eval_runs}x{args.final_eval_episodes}")
    if wandb and args.wandb:
        wandb.log({
            "final/eval_mean": final_mean,
            "final/eval_std": final_std,
            "final/runs": args.final_eval_runs,
            "final/episodes": args.final_eval_episodes
        })
        wandb.finish()
