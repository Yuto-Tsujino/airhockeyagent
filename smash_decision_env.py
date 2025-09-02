from __future__ import annotations
import math
import os
from baseline.baseline_agent.baseline_agent import BaselineAgent
from baseline.baseline_agent.system_state import SystemState
from baseline.baseline_agent.trajectory_generator import TrajectoryGenerator
import numpy as np

# 先頭付近
try:
    import gymnasium as gym
    import gymnasium.spaces as spaces
except Exception:
    import gym
    import gym.spaces as spaces


class SmashDecisionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
            self,
            env_info: dict,
            agent_params: dict | None = None,
            seed: int | None = None,
            use_baseline: bool = False
    ) -> None:
        super().__init__()
        self.env_info = env_info
        self.agent_params = dict(agent_params or {})
        self.rng = np.random.default_rng(seed)
        self.use_baseline = bool(use_baseline)

        # 物理・テーブル寸法のキャッシュ
        table = env_info["table"]
        self.table_w = float(table["width"])
        self.table_l = float(table["length"])
        self.goal_w = float(table.get("goal_width", self.table_w * 0.5))
        self.x_offset = float(env_info["table"]["x_offset"] if "x_offset" in env_info["table"] else 0.0)

        self.dt = 1.0 / float(env_info["robot"]["control_frequency"])

        self.obs_dim = 7

        obs_low  = -np.ones(self.obs_dim, dtype=np.float32)
        obs_high =  np.ones(self.obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.dy_max = float(self.goal_w * 0.5 * 0.9)
        a_low = np.array([-self.dy_max, 0.6, 0.3], dtype=np.float32)
        a_high = np.array([+self.dy_max, 1.0, 0.9], dtype=np.float32)
        self.action_space = spaces.Box(low=a_low, high=a_high, dtype=np.float32)


        # 内部状態
        self._px = 0.0
        self._py = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._pred_x = 0.0
        self._pred_y = 0.0
        self._T = 0.0
        self._H = 1.0
        self._done = False
        self._step_count = 0
        self.max_episode_steps = int(self.agent_params.get("max_episode_steps", 200))

        if self.use_baseline:
            # Baseline構築
            tmp = BaselineAgent(self.env_info)
            base_params = dict(tmp.agent_params)
            base_params.update(self.agent_params)
            self.agent_params = base_params

            self.state = SystemState(self.env_info, agent_id=1, agent_params=self.agent_params)
            self.traj_gen = TrajectoryGenerator(self.env_info, self.agent_params, self.state)

            # # 初期姿勢（ゼロでもOK。必要に応じて x_home を使って初期化）
            # self.state.q_cmd[:] = 0.0
            # self.state.dq_cmd[:] = 0.0
            # self.state.x_cmd[:] = 0.0
            # self.state.v_cmd[:] = 0.0

            home_q = np.asarray(self.agent_params['joint_anchor_pos'], dtype=np.float32)
            self.state.q_cmd[:] = home_q
            self.state.dq_cmd[:] = 0.0

            self.state.x_cmd, self.state.v_cmd = self.state.update_ee_pos_vel(self.state.q_cmd, self.state.dq_cmd)
            # x_home = np.asarray(self.agent_params)

        self._dbg_on = os.getenv("AH_CHK", "") == "1"
        self._dbg_n  = 0         # 出力した行数
        self._dbg_max = 200      # 出し過ぎ防止（適宜）
        if self._dbg_on:
            print("[CHK] debug ON")




    def _sample_event_state(self):
        # _sample_event_state() 内で self._px を決めている箇所をこれで上書き
        lo, hi = map(float, self.agent_params.get('hit_range', [0.8, 1.3]))
        margin = 0.05  # 端を少し避ける
        px_lo = lo + margin
        px_hi = hi - margin
        self._px = float(np.random.uniform(px_lo, px_hi))
        self._py = float(self.rng.normal(0.0, 0.05))
        self._vx = float(self.rng.normal(0.0, 0.03))
        self._vy = float(self.rng.normal(0.0, 0.03))

        # 簡易予測
        self._H = 1.0
        self._T = float(self.rng.uniform(0.4, 0.8))
        #予測位置は等速直線で近似
        self._pred_x = self._px + self._vx * self._T
        self._pred_y = self._py + self._vy * self._T
        if self._dbg_on and self._dbg_n < self._dbg_max:
            lo, hi = map(float, self.agent_params.get('hit_range', [0.8, 1.3]))
            S = (float(self.x_offset) - self._px) * self._vx
            print(f"[EVT] px={self._px:.3f}, py={self._py:.3f}, vx={self._vx:.3f}, vy={self._vy:.3f},"
                f" T={self._T:.2f}, H={self._H:.2f}, in_hit_range={lo<=self._px<=hi}, S={S:.4f}")
            self._dbg_n += 1

    def _build_obs(self):
        return self._build_policy_obs()


    def _optimize_joint_traj(self, cart_traj):
        if cart_traj is None or not hasattr(cart_traj, "size") or cart_traj.size == 0:
            return []
        success, traj = self.traj_gen.optimize_trajectory(
            cart_traj, self.state.q_cmd, self.state.dq_cmd, self.agent_params['joint_anchor_pos']
        )
        if not success or traj is None or len(traj) == 0:
            return []
        return list(traj)

    def _build_bezier(self, start_xy, start_v, target_xy, target_v, tfin, max_steps=30):
        self.traj_gen.bezier_planner.compute_control_point(
            np.asarray(start_xy, np.float32), np.asarray(start_v, np.float32),
            np.asarray(target_xy, np.float32), np.asarray(target_v, np.float32), float(tfin)
        )
        steps_needed = int(np.ceil(float(tfin) / self.dt)) + 1
        steps = int(np.clip(steps_needed, 10, 80))
        return self.traj_gen.generate_bezier_trajectory(max_steps=steps)
    
    # smash_decision_env.py 内のクラスに追加
    def _build_policy_obs(self) -> np.ndarray:
        # スケール定義
        x_span = max(self.table_l * 0.5 - float(self.x_offset), 1e-6)  # stop_line→センターまで
        y_span = max(self.table_w * 0.5, 1e-6)
        v_scale = float(self.agent_params.get("puck_v_max", 3.0))

        # 位置 [-1,1]
        px01 = (self._px - float(self.x_offset)) / x_span                # 0..1 目安
        px_n  = float(np.clip(2.0 * px01 - 1.0, -1.0, 1.0))
        py_n  = float(np.clip(self._py / y_span, -1.0, 1.0))

        # 速度 [-1,1]
        vx_n = float(np.clip(self._vx / v_scale, -1.0, 1.0))
        vy_n = float(np.clip(self._vy / v_scale, -1.0, 1.0))

        # 時間特徴
        H = float(self._H)
        T = float(self._T)
        TH = float(np.clip(T / max(H, 1e-6), 0.0, 1.0))                  # 0..1
        TH_n = float(2.0 * TH - 1.0)                                     # [-1,1]
        H_n  = float(np.clip(H / 1.0, 0.0, 1.0))                         # だいたい 1.0 付近

        # ゲートの向き（符号）
        S_gate = (float(self.x_offset) - self._px) * self._vx
        S_n = float(np.tanh(10.0 * S_gate))                              # [-1,1] に圧縮

        obs = np.array([px_n, py_n, vx_n, vy_n, TH_n, H_n, S_n], dtype=np.float32)
        return obs


    #Gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._step_count = 0
        self._done = False
        self._sample_event_state()
        return self._build_obs(), {}
    
    def step(self, action: np.ndarray):
        steps = int(self.agent_params.get("max_plan_steps", 10))
        T_min = max(2*self.dt, 2*steps*self.dt)
        stop_line = float(self.x_offset)
        S = (stop_line - self._px) * self._vx
        event_ok = (S > -1e-3) and (self._T >= T_min) and (self._T < self._H - 1e-3)
        # def step(self, action): の中、event_ok を計算した直後
        if self._dbg_on and self._dbg_n < self._dbg_max:
            S_gate = (float(self.x_offset) - self._px) * self._vx
            print(f"[GT ] event_ok={event_ok}, S_gate={S_gate:.4f}, T={self._T:.2f}, T_min={T_min:.2f}, H={self._H:.2f}")
            print(f"[ENV] use_baseline={self.use_baseline}")

            self._dbg_n += 1

        if self._done:
            return self._build_obs(), 0.0, True, False, {}
        
        self._step_count += 1


        # 行動を枠にクリップ
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        dY, alpha, beta = float(action[0]), float(action[1]), float(action[2])

        truncated = (self._step_count >= self.max_episode_steps)

        if not event_ok:
            # 行動は無視。報酬0で次のイベントへ
            self._sample_event_state()
            return self._build_obs(), 0.0, False, truncated, { "event_ok": False }
        
        if not self.use_baseline:
            align = max(0.0, 1.0 - abs(self._pred_y + dY) / (self.table_w * 0.5))
            time_ok = 1.0 if (0.2 <= self._T <= 0.9) else 0.0
            reward = 0.2 * align + 0.1 * time_ok + 0.2 * (alpha - 0.6) / 0.4
            
            self._sample_event_state()      
        
            return self._build_obs(), float(reward), False, truncated, {
                "event_ok": True,
                "opt_ok": False,
                "align": align,
                "time_ok": time_ok,
                "alpha": alpha,
                "beta": beta,
            }
        # ---- 本番：Baseline 連携（ベジエ→最適化）----
        # 打点Yを安全枠でクリップ
        y_lim = self.table_w * 0.5 - self.env_info['mallet']['radius'] - 0.02
        goal_y = float(np.clip(self._py + dY, -y_lim, y_lim))
        goal_pos = np.array([self.table_l / 2.0, goal_y], dtype=np.float32)

        # パック→ゴール方向ベクトル
        puck = np.array([self._px, self._py], dtype=np.float32)
        dir_vec = goal_pos - puck
        n = float(np.linalg.norm(dir_vec))
        if not np.isfinite(n) or n < 1e-8:
            self._sample_event_state()
            return self._build_obs(), -0.5, False, truncated, {
                "event_ok": True, "opt_ok": False, "align": 0.0,
                "alpha": alpha, "beta": beta
            }
        dir_vec /= n

        # 打速
        v_min = 0.2
        v_max = float(self.agent_params.get('max_hit_velocity', 3.0))
        hit_v = float(np.clip(v_min * (alpha ** 2) * (v_max - v_min), v_min, v_max))
        hit_vel_2d = dir_vec * hit_v

        backoffs = [0.02, 0.04]
        best_traj = None
        best_align = 0.0
        best_opt_ok = False

        for bo in backoffs:
            hit_pos_2d = puck - dir_vec * (float(self.env_info['mallet']['radius']) + bo)

            # 到達時刻
            T_plan = float(np.clip(beta * (self._H - 1e-3), T_min, self._H - 1e-3))

            # ベジエ→最適化
            start_xy = self.state.x_cmd[:2].astype(np.float32)
            start_v  = self.state.v_cmd[:2].astype(np.float32)
            cart = self._build_bezier(start_xy, start_v, hit_pos_2d, hit_vel_2d, T_plan)
            if self._dbg_on and self._dbg_n < self._dbg_max and cart is not None and hasattr(cart, "shape"):
                p = cart[:, 0:2]  # 位置 (N,2)
                T_inf = (p.shape[0] - 1) * self.dt
                dx = float(p[-1, 0] - p[0, 0])
                dy = float(p[-1, 1] - p[0, 1])
                vavg_x = abs(dx) / max(T_inf, 1e-6)
                print(f"[BZ ] N={p.shape[0]}, T={T_inf:.2f}, Δx={dx:.3f}, Δy={dy:.3f}, vavg_x={vavg_x:.2f}")
                self._dbg_n += 1
            traj = self._optimize_joint_traj(cart)

            align = max(0.0, 1.0 - abs(self._pred_y + dY) / (self.table_w * 0.5))

            if traj:
                best_traj = traj
                best_align = align
                best_opt_ok = True
                break
            else:
                if not best_opt_ok:
                    best_align = align

        if best_opt_ok:
            reward = 1.0 + 0.2 * best_align
            q_last, dq_last = best_traj[-1]
            self.state.q_cmd = q_last.copy()
            self.state.dq_cmd = dq_last.copy()
            self.state.x_cmd, self.state.v_cmd = self.state.update_ee_pos_vel(self.state.q_cmd, self.state.dq_cmd)
        else:
            reward = -0.5 + 0.2 * best_align

        self._sample_event_state()
        return self._build_obs(), float(reward), False, truncated, {
            "event_ok": True, "opt_ok": best_opt_ok, "align": align,
            "alpha": alpha, "beta": beta
        }

    
    def render(self):
        return None
    
    def close(self):
        return None
