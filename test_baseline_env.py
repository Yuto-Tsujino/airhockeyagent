import pickle
import os
import numpy as np
from air_hockey_agent.smash_decision_env import SmashDecisionEnv

# 1) 本物の env_info をロード
with open(r"C:\Airhockeychallenge2025\env_info.pkl", "rb") as f:
    env_info = pickle.load(f)

# 2) Baseline 連携モードで起動
env = SmashDecisionEnv(
    env_info,
    {"max_plan_steps": 8, "stop_line_x": 0.0, "learn_dy_max": 0.05, "min_hit_velocity": 0.2},
    seed=0,
    use_baseline=True
)

# 3) 一歩進めて統計を見る（軽いスモーク）
obs, info = env.reset()
N = 100
event_ok_cnt = 0
opt_ok_cnt = 0
print("obs_space:", env.observation_space.shape)
printed = False
for _ in range(N):
    a = env.action_space.sample()        # まだ学習前なのでランダム
    obs, r, term, trunc, info = env.step(a)
    if not printed:
        print("obs shape:", obs.shape, "min/max:", float(obs.min()), float(obs.max()))
        printed = True
    if info.get("event_ok", False):
        event_ok_cnt += 1
        if info.get("opt_ok", False):
            opt_ok_cnt += 1

print("event_ok ratio =", event_ok_cnt / N)
print("opt_ok / event_ok =", (opt_ok_cnt / event_ok_cnt) if event_ok_cnt else 0.0)
