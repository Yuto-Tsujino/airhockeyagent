import os, time, argparse, pickle
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

from air_hockey_agent.smash_decision_env import SmashDecisionEnv

def make_env(seed=0):
    # テスト用に使っていた簡易 env_info と同等でOK（SmashDecisionEnv側で不足キーは補完）
    ENV_INFO_PATH = os.environ.get(
        "ENV_INFO_PATH",
        os.path.join(os.path.dirname(__file__), "..", "env_info.pkl")
    )
    with open(ENV_INFO_PATH, "rb") as f:
        env_info = pickle.load(f)

    agent_params = {
        "max_plan_steps": 10,
        "learn_dy_max": 0.10,     # 学習する dY の最大値（必要なら 0.05→0.10 など調整）
        "use_baseline": True,     # Baseline連携で報酬を出す（本番想定）
    }
    env = SmashDecisionEnv(env_info, agent_params, seed=seed, use_baseline=True)
    return env

def eval_once(model, n_steps=2000, seed=123):
    env = make_env(seed=seed)
    obs, info = env.reset()
    ep_r, R, event_ok_cnt, ok_and_opt_cnt = 0.0, 0.0, 0, 0
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        R += r
        if info.get("event_ok", False):
            event_ok_cnt += 1
            if info.get("opt_ok", False):
                ok_and_opt_cnt += 1
        if term or trunc:
            obs, info = env.reset()
    ratio = (ok_and_opt_cnt / max(event_ok_cnt,1))
    return R / n_steps, event_ok_cnt / n_steps, ratio

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--logdir", type=str, default="runs/smash_sac_v0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    env = DummyVecEnv([lambda: make_env(seed=args.seed)])
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=200_000,
        tau=0.02,
        gamma=0.99,
        train_freq=64,
        gradient_steps=64,
        ent_coef="auto",
        seed=args.seed,
        tensorboard_log=args.logdir
    )
    logger = configure(args.logdir, ["stdout", "tensorboard"])
    model.set_logger(logger)

    print("[train] start")
    model.learn(total_timesteps=args.steps, log_interval=10, progress_bar=True)

    save_path = os.path.join(args.logdir, "sac_air_hockey_attacker_final.zip")
    model.save(save_path)
    print(f"[train] saved: {save_path}")
    export_path = os.path.join(os.path.dirname(__file__), "sac_air_hockey_attacker_final.zip")
    model.save(export_path)
    print(f"[train] exported for MyAgent: {export_path}")

    # かんたん自己評価
    avg_r, event_rate, opt_hit_rate = eval_once(model, n_steps=3000, seed=args.seed+1)
    print(f"[eval] avg_r/step={avg_r:.3f}, event_ok_rate/step={event_rate:.3f}, opt_ok|event={opt_hit_rate:.3f}")

if __name__ == "__main__":
    # 学習中のデバッグ出力は邪魔なのでOFF推奨
    os.environ.pop("AH_CHK", None)
    main()
