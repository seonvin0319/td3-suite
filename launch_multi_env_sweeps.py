# File: launch_multi_env_sweeps.py
import os
import math
import copy
import wandb

PROJECT = "gflow_offrl"
ENTITY  = "seonvin0319"

# 이 프로세스가 직접 돌릴 에이전트 런 수(옵션: 여러 터미널에서 병렬 권장)
AGENTS_PER_SWEEP = int(os.environ.get("AGENTS_PER_SWEEP", "2"))

# ★ 환경군별 자동 종료 예산(run_cap). 필요시 env var로 조정
MUJOCO_CAP  = int(os.environ.get("RUN_CAP_MUJOCO", "30"))   # hopper/halfcheetah/walker
ANTMAZE_CAP = int(os.environ.get("RUN_CAP_ANTMAZE", "40"))  # antmaze 계열
DEFAULT_CAP = int(os.environ.get("RUN_CAP_DEFAULT", "30"))

# ★ 환경군별 '최소 스텝 후에만 프루닝' (필요시 env var로 조정)
MUJOCO_MIN_STEPS  = int(os.environ.get("MIN_STEPS_BEFORE_PRUNE_MUJOCO", "100000"))  # 100k
ANTMAZE_MIN_STEPS = int(os.environ.get("MIN_STEPS_BEFORE_PRUNE_ANTMAZE", "250000")) # 250k

def _cap_for_env(env_name: str) -> int:
    if "antmaze" in env_name:
        return ANTMAZE_CAP
    return MUJOCO_CAP

def _min_steps_for_env(env_name: str) -> int:
    if "antmaze" in env_name:
        return ANTMAZE_MIN_STEPS
    return MUJOCO_MIN_STEPS

# 공통 sweep 설정 (alpha=1 고정, w2_weight 중심 튜닝)
BASE_SWEEP = {
    "program": "main.py",
    "project": PROJECT,
    "entity": ENTITY,
    "method": "bayes",
    "metric": {"name": "final/eval_mean", "goal": "maximize"},
    # early_terminate는 env별로 make_sweep_for_env에서 주입/오버라이드함
    "parameters": {
        "policy":        {"value": "GFlow_W2"},
        "normalize":     {"value": True},
        "max_timesteps": {"value": 500000},
        "eval_freq":     {"value": 5000},     # ← 5k마다 평가
        "alpha":         {"value": 1.0},

        # 고정값(요청대로)
        "seed":          {"values": [0,1]},
        "batch_size":    {"value": 256},

        # 핵심 튜닝 파라미터
        "w2_weight": {
            "distribution": "log_uniform_values",  # 값 범위 그대로 사용
            "min": 0.05, "max": 3.0
        },
        "entropy_weight": {"values": [0.0, 0.005, 0.01]},
        "policy_freq": {"value": 1},

        # 최종 평가 설정
        "final_eval_runs":     {"value": 5},
        "final_eval_episodes": {"value": 10},

        # 라우팅(스윕 모드에선 보통 무시되지만 넣어도 무해)
        "wandb_project": {"value": PROJECT},
        "wandb_entity":  {"value": ENTITY},
    },
    # 불리언 플래그는 직접, 나머지는 ${args}로 안전 전개
    "command": [
        "${env}", "${interpreter}", "${program}",
        "--wandb",
        "${args}"
    ],
}

# ▶︎ 여기서 원하는 env들을 선언 (D4RL 버전에 맞게 v0/v2 확인)
HOPPER_ENVS = [
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-medium-expert-v2",
]
HALFCHEETAH_ENVS = [
    "halfcheetah-medium-v2", "halfcheetah-medium-replay-v2", "halfcheetah-medium-expert-v2",
]
WALKER_ENVS = [
    "walker2d-medium-v2", "walker2d-medium-replay-v2", "walker2d-medium-expert-v2",
]
ANTMAZE_ENVS = [
    "antmaze-umaze-v2", "antmaze-umaze-diverse-v2",
    "antmaze-medium-play-v2", "antmaze-medium-diverse-v2",
    "antmaze-large-play-v2",  "antmaze-large-diverse-v2",
]

ALL_GROUPS = [
    ("hopper", HOPPER_ENVS),
    ("halfcheetah", HALFCHEETAH_ENVS),
    ("walker2d", WALKER_ENVS),
    ("antmaze", ANTMAZE_ENVS),
]

def make_sweep_for_env(env_name: str) -> str:
    """단일 env에 대한 스윕 생성 + run_cap/early_terminate 주입"""
    cfg = copy.deepcopy(BASE_SWEEP)
    cfg["parameters"]["env"] = {"value": env_name}
    cfg["run_cap"] = _cap_for_env(env_name)  # ★ 자동 종료 런 수

    # ── 환경별 '최소 스텝 보장' 기반으로 min_iter 계산 ──
    eval_freq = int(cfg["parameters"]["eval_freq"]["value"])
    max_ts    = int(cfg["parameters"]["max_timesteps"]["value"])
    total_iter = max(1, max_ts // eval_freq)  # 전체 평가 횟수 (ex. 500k/5k=100)

    min_steps = _min_steps_for_env(env_name)  # MuJoCo=100k, AntMaze=250k
    min_iter  = math.ceil(min_steps / eval_freq)       # ex. 100k/5k=20, 250k/5k=50
    min_iter  = min(min_iter, total_iter)              # 보호: min_iter ≤ total_iter

    s_val = 2 if "antmaze" in env_name else 1
    cfg["early_terminate"] = {"type": "hyperband", "min_iter": int(min_iter), "s": int(s_val)}

    sweep_id = wandb.sweep(cfg, project=PROJECT, entity=ENTITY)
    return sweep_id

def launch_agent(sweep_id: str, count: int):
    wandb.agent(sweep_id, project=PROJECT, entity=ENTITY, count=count)

if __name__ == "__main__":
    created = []
    for group_name, envs in ALL_GROUPS:
        for env in envs:
            sid = make_sweep_for_env(env)
            cap = _cap_for_env(env)
            min_steps = _min_steps_for_env(env)
            print(f"[SWEEP CREATED] {group_name}/{env} -> {sid}  (run_cap={cap}, min_steps_before_prune={min_steps})")
            created.append((env, sid))

    print("\n=== To run agents in separate terminals ===")
    for env, sid in created:
        print(f"wandb agent {ENTITY}/{PROJECT}/{sid}")

    RUN_NOW = int(os.environ.get("RUN_NOW", "0"))
    if RUN_NOW:
        for env, sid in created:
            print(f"\n[RUNNING] {env} :: {sid}  (count={AGENTS_PER_SWEEP})")
            launch_agent(sid, count=AGENTS_PER_SWEEP)
