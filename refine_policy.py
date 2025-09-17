# File: td3_suite/td3w/refine_policy.py

import argparse
import os
import copy
import numpy as np
import torch
import gym
import d4rl

import utils
import agent  # TD3_BC
try:
    from agent_gflow import TD3_GFlow_BC 
except Exception:
    TD3_GFlow_BC = None

# -----------------------------
# eval (D4RL normalized score)
# -----------------------------
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    env = gym.make(env_name)
    env.seed(seed + seed_offset)
    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, done = env.reset(), False
        while not done:
            state_n = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(state_n)
            state, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = env.get_normalized_score(avg_reward) * 100.0
    return float(avg_reward), float(d4rl_score)


# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3_BC", choices=["TD3_BC", "TD3_GFlow_BC"])
    parser.add_argument("--env", default="hopper-medium-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000000, help="actor-only refinement steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="tiny lr for safe refinement")

    # TD3 core hparams (for building the agent consistently)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_freq", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=2.5)

    # Loss weights for refinement
    parser.add_argument("--bc_weight", type=float, default=1.8, help="only used by TD3_BC refinement")
    parser.add_argument("--w2_weight", type=float, default=0.5, help="only used by TD3_GFlow_BC")
    parser.add_argument("--entropy_weight", type=float, default=0.01, help="only used by TD3_GFlow_BC")

    # data & eval
    parser.add_argument("--normalize", type=str, default="True")
    parser.add_argument("--eval_freq", type=int, default=5e3)
    parser.add_argument("--eval_episodes", type=int, default=10)

    # io
    parser.add_argument("--models_dir", default="./models")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--load_model", required=True,
                        help="base filename used when saving: e.g., TD3_BC_hopper-medium-v2_0")
    parser.add_argument("--save_suffix", default="_refined",
                        help="suffix to append when saving refined actor")
    parser.add_argument("--load_actor_from", default=None, help="(optional) path prefix to an actor checkpoint to initialize actor from. "
         "e.g., ./models/TD3_GFlow_BC_hopper-medium-v2_0_refined_actor")

    # wandb (optional)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="td3_suite")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--run_name", default="")

    args = parser.parse_args()

    # ----------------- setup -----------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

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

    # ----------------- agent init & load -----------------
    if args.policy == "TD3_BC":
        policy = agent.TD3_BC(**kwargs)
    else:
        assert TD3_GFlow_BC is not None, "agent_gflow.py 가 필요합니다."
        policy = TD3_GFlow_BC(**kwargs)

    load_base = os.path.join(args.models_dir, args.load_model)
    policy.load(load_base)  # 이 함수는 클래스별로 *_critic, *_actor 등 여러 파일을 읽음
    
    if args.load_actor_from is not None:
        actor_sd = torch.load(args.load_actor_from)  # full path WITHOUT suffixes; here it's the exact file
        policy.actor.load_state_dict(actor_sd)
        policy.actor_target = copy.deepcopy(policy.actor)
        print(f"[refine] Initialized ACTOR from: {args.load_actor_from}")


    # ----------------- freeze others -----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.actor.train()
    for p in policy.actor.parameters(): p.requires_grad_(True)

    # critic/targets 고정
    for p in policy.critic.parameters(): p.requires_grad_(False)
    for p in policy.critic_target.parameters(): p.requires_grad_(False)
    if hasattr(policy, "actor_target"):
        for p in policy.actor_target.parameters(): p.requires_grad_(False)
    # behavior policy 고정 (GFlow)
    if hasattr(policy, "behavior_policy"):
        for p in policy.behavior_policy.parameters(): p.requires_grad_(False)

    # optimizer
    opt = torch.optim.Adam(policy.actor.parameters(), lr=args.lr)

    # ----------------- data buffer -----------------
    rb = utils.ReplayBuffer(state_dim, action_dim)
    rb.convert_D4RL(d4rl.qlearning_dataset(env))
    if args.normalize.lower() in ["true", "1", "yes", "y"]:
        mean, std = rb.normalize_states()
    else:
        mean, std = 0, 1

    # ----------------- wandb -----------------
    run = None
    if args.wandb:
        try:
            import wandb
            run_name = args.run_name if args.run_name else f"refine_{args.env}_{args.policy}_seed{args.seed}"
            init_kwargs = dict(
                project=args.wandb_project, name=run_name, group=args.env,
                tags=[args.policy, "refine"], config=vars(args)
            )
            if args.wandb_entity:
                init_kwargs["entity"] = args.wandb_entity
            run = wandb.init(**init_kwargs)
        except Exception as e:
            print(f"[wandb] init failed: {e}")
            run = None
            args.wandb = False

    # ----------------- helper: compute actor loss -----------------
    def actor_loss_td3bc(states, actions):
        # Q term
        with torch.no_grad():
            pass
        pi = policy.actor(states)
        Q = policy.critic.Q1(states, pi)
        lam = args.alpha / (Q.abs().mean().detach() + 1e-8)
        # BC
        bc = torch.nn.functional.mse_loss(pi, actions)
        loss = args.bc_weight * bc -lam * Q.mean()
        return loss, dict(Q_mean=float(Q.mean().item()),
                          lambda_=float(lam.item()),
                          bc=float(bc.item()))

    def actor_loss_gflow(states, actions):
        # mean/logstd from actor
        mean_a, logstd_a = policy.actor(states)
        Q = policy.critic.Q1(states, mean_a)
        lam = args.alpha / (Q.abs().mean().detach() + 1e-8)

        # Wasserstein to frozen behavior
        with torch.no_grad():
            mean_b, logstd_b = policy.behavior_policy(states)
            std_b = torch.exp(logstd_b)
        std_a = torch.exp(logstd_a)
        w2 = ((mean_a - mean_b) ** 2).mean() + ((std_a - std_b) ** 2).mean()

        # entropy (maximize)
        entropy = 0.5 * (2 * logstd_a + np.log(2 * np.pi * np.e)).sum(dim=1).mean()

        loss = args.w2_weight * w2 - lam * Q.mean() - args.entropy_weight * entropy
        return loss, dict(Q_mean=float(Q.mean().item()),
                          lambda_=float(lam.item()),
                          w2=float(w2.item()),
                          entropy=float(entropy.item()))

    # ----------------- before refinement eval -----------------
    pre_avg, pre_d4rl = eval_policy(policy, args.env, args.seed, mean, std, eval_episodes=args.eval_episodes)
    print(f"[refine] Before: avg_reward={pre_avg:.2f}, d4rl={pre_d4rl:.2f}")
    if args.wandb:
        wandb.log({"eval/before_avg_reward": pre_avg, "eval/before_d4rl": pre_d4rl}, step=0)

    # ----------------- refinement loop (actor only) -----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for t in range(1, args.steps + 1):
        states, actions, _, _, _ = rb.sample(args.batch_size)
        # states는 normalize된 버퍼를 썼으므로 그대로 사용

        if args.policy == "TD3_BC":
            loss, extras = actor_loss_td3bc(states, actions)
        else:
            loss, extras = actor_loss_gflow(states, actions)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), max_norm=5.0)
        opt.step()

        if args.wandb and t % 50 == 0:
            log = {"refine/actor_loss": float(loss.item()), "step": t}
            log.update({f"refine/{k}": v for k, v in extras.items()})
            wandb.log(log, step=t)

        if t % args.eval_freq == 0 or t == args.steps:
            avg_r, dscore = eval_policy(policy, args.env, args.seed, mean, std, eval_episodes=args.eval_episodes)
            print(f"[refine] step {t}/{args.steps} | avg_reward={avg_r:.2f}, d4rl={dscore:.2f}")
            if args.wandb:
                wandb.log({"eval/avg_reward": avg_r, "eval/d4rl": dscore, "step": t}, step=t)

    # ----------------- after refinement save -----------------
    save_base = os.path.join(args.models_dir, args.load_model + args.save_suffix)
    policy.save(save_base)
    print(f"[refine] Saved refined actor to: {save_base}_actor (and related files)")

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
