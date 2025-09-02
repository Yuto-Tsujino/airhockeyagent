import os, pickle
import numpy as np
from stable_baselines3 import SAC

from air_hockey_challenge.framework import AgentBase
from baseline.baseline_agent.baseline_agent import BaselineAgent
from baseline.baseline_agent.system_state import SystemState
from baseline.baseline_agent.trajectory_generator import TrajectoryGenerator


LOG = bool(int(os.getenv("AH_LOG", "0")))


class MyAgent(AgentBase):
    def __init__(self, env_info, agent_id=1, **kwargs):
        super(MyAgent, self).__init__(env_info, agent_id, **kwargs)

        # 既定パラメータ（Baselineから継承して最小限上書き）
        self.baseline = BaselineAgent(self.env_info)
        self.agent_params = dict(self.baseline.agent_params)
        
        self.state: SystemState = SystemState(self.env_info, agent_id, self.agent_params)
        self.traj_generator: TrajectoryGenerator = TrajectoryGenerator(self.env_info, self.agent_params, self.state)

        self.dt = 1.0 / float(self.env_info['robot']['control_frequency'])

        #SACモデルロード
        model_env = os.getenv("AH_SAC_MODEL", "").strip()
        candidates = []
        if model_env:
            # そのまま / CWD 相対 / MyAgent 相対 の順に試す
            candidates += [
                model_env,
                os.path.join(os.getcwd(), model_env),
                os.path.join(os.path.dirname(__file__), model_env),
            ]
        default_name = "sac_air_hockey_attacker_final.zip"
        candidates += [
            os.path.join(os.path.dirname(__file__), default_name),
            os.path.join(os.getcwd(), default_name)
        ]

        self.sac_model = None
        for p in candidates:
            if os.path.exists(p):
                self.sac_model = SAC.load(p, device="cpu")
                if LOG:
                    print(f"[MyAgent] SAC model loaded: {os.path.basename(p)}")
                break
        if self.sac_model is None:
            print("[MyAgent] SAC model not found. Fallback to baseline-only")

        # SAC出力のスケール
        self.delta_y_limit = 0.05
        self.alpha_bounds = (0.6, 1.0)

        # 内部バッファの初期化
        self.state.trajectory_buffer = []

        self.goal_width   = float(self.env_info['table']['goal_width'])
        self.table_length = float(self.env_info['table']['length'])

        self._frame = 0
        self._log_every = int(os.getenv("AH_LOG_EVERY", "200"))
        self._cnt_sac_try = 0
        self._cnt_sac_ok  = 0
        self._cnt_fb      = 0
        self._cnt_base    = 0
        self._cnt_gate_ng = 0

        self.agent_params.setdefault('max_plan_steps', 10) 

        self.enable_sac = bool(int(os.getenv("AH_ENABLE_SAC", "1")))

        if os.getenv("DUMP_ENV_INFO", "") == "1":
            out = os.getenv("ENV_INFO_PATH", "env_info.pkl")
            try:
                with open(out, "wb") as f:
                    pickle.dump(self.env_info, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"[MyAgent] dumped env_info -> {out}")
            except Exception as e:
                print(f"[MyAgent] dump failed: {e}")
        # joint_anchor_pos = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.])
        # x_home = np.array([0.65, 0., self.env_info['robot']['ee_desired_height']])
        
    def reset(self):
        self.state.reset()
        self.new_start = True

    # ---------- 内部ユーティリティ ----------
    def _optimize_joint_traj(self, cart_traj):
        """ベジエのデカルト軌道 -> 関節軌道最適化。成功なら list[(q,dq),...] を返す"""
        if cart_traj is None or not hasattr(cart_traj, "size") or cart_traj.size == 0:
            return []
        success, traj = self.traj_generator.optimize_trajectory(
            cart_traj, self.state.q_cmd, self.state.dq_cmd, self.agent_params['joint_anchor_pos']
        )
        if not success or traj is None or len(traj) == 0:
            return []
        return list(traj)

    # MyAgent.py 内 _build_bezier を差し替え
    def _build_bezier(self, start_xy, start_v, target_xy, target_v, tfin, max_steps=30):
        self.traj_generator.bezier_planner.compute_control_point(
            np.asarray(start_xy, dtype=np.float32),
            np.asarray(start_v, dtype=np.float32),
            np.asarray(target_xy, dtype=np.float32),
            np.asarray(target_v, dtype=np.float32),
            float(tfin)
        )
        steps_needed = int(np.ceil(float(tfin) / self.dt)) + 1
        steps = int(np.clip(steps_needed, 10, 80))
        return self.traj_generator.generate_bezier_trajectory(max_steps=steps)

    # def _sac_hit_goal(self, observation):
    #     """SACの2D出力を(打点,打速)にデコードして目標を返す"""
    #     obs = self._build_policy_obs()
    #     act2, _ = self.sac_model.predict(obs, deterministic=True)  # [y_norm, power_norm]
    #     a = np.asarray(act2, dtype=np.float32).reshape(-1)

    #     goal_y = float(a[0]) * (self.goal_width * 0.5 * 0.9)
    #     hit_v = (float(a[1]) + 1.0) * 0.5 * (self.agent_params['max_hit_velocity'] - 0.5) + 0.5

    #     puck = self.get_puck_pos(observation)[:2].astype(np.float32)
    #     goal_pos = np.array([self.table_length / 2, goal_y], dtype=np.float32)

    #     dir_vec = goal_pos - puck
    #     n = float(np.linalg.norm(dir_vec))
    #     if not np.isfinite(n) or n < 1e-8:
    #         return None  # 方向が作れない
    #     dir_vec /= n

    #     hit_vel_2d = dir_vec * hit_v
    #     hit_pos_2d = puck - dir_vec * float(self.env_info['mallet']['radius'])  # パックの直前を狙う
    #     return hit_pos_2d, hit_vel_2d

    def _try_sac_plan(self, observation):
        """SACを使った打撃を複数条件で試す。成功したらtraj(list)を返す。"""
        self.state.update_prediction(self.agent_params.get('max_prediction_time', 1.0))

        H = float(getattr(self.state, "prediction_horizon", 1.0))
        T = float(getattr(self.state, "predicted_time", 0.0))
        steps = int(self.agent_params.get('max_plan_steps', 10))
        T_min = max(2 * self.dt, 2 * steps * self.dt)

        x_offset = float(self.env_info["table"].get("x_offset", 0.0))
        stop_line = float(self.agent_params.get('stop_line_x', x_offset))
        px = float(self.state.estimated_state[0]); vx = float(self.state.estimated_state[2])
        S =(stop_line - px) * vx
        
        if not (S > -1e-3) and (T >= T_min) and (T < H - 1e-3):
            self._cnt_gate_ng += 1
            if LOG and (self._frame % self._log_every == 0):
                print(f"[GATE] block S={S:.4f}  T={T:.2f}  T_min={T_min:.2f}  H={H:.2f}")
            return []
        
        self._cnt_sac_try += 1
        
        if self.sac_model is None:
            return []
        
        #3次元アクションの取得(7次元)
        obs7 = self._build_policy_obs()
        act, _ = self.sac_model.predict(obs7, deterministic=True)
        dY, alpha, beta = float(act[0]), float(act[1]), float(act[2])

        alpha = float(np.clip(alpha, 0.6, 1.0))
        beta = float(np.clip(beta, 0.3, 0.9))

        if LOG and (self._frame % self._log_every == 0):
            print(f"[ACT] dY={dY:.3f} alpha={alpha:.3f} beta={beta:.3f}")

        table_w = float(self.env_info["table"]["width"])
        y_lim = table_w * 0.5 - self.env_info["mallet"]["radius"] - 0.02
        puck = self.get_puck_pos(observation)[:2].astype(np.float32)
        goal_y = float(np.clip(puck[1] + dY, -y_lim, y_lim))
        goal_pos = np.array([self.table_length / 2.0, goal_y], dtype=np.float32)

        # ゴールの方向
        dir_vec = goal_pos - puck
        n = float(np.linalg.norm(dir_vec))
        if not np.isfinite(n) or n < 1e-8:
            return []
        dir_vec /= n

        # 打速
        v_min = 0.2
        v_max = float(self.agent_params.get("max_hit_velocity", 3.0))
        hit_v = float(np.clip(v_min + (alpha - self.alpha_bounds[0]) / (self.alpha_bounds[1] - self.alpha_bounds[0]) * (v_max - v_min),
                          v_min, v_max))
        hit_vel_2d = dir_vec * hit_v

        # 到達時刻
        T_plan = float(np.clip(beta * (H - 1e-3), T_min, H - 1e-3))

        if LOG and (self._frame % self._log_every == 0):
            print(f"[GATE] S={S:.4f}  T={T:.2f}  T_min={T_min:.2f}  H={H:.2f}")

        # ベジエ→最適化
        start_xy = self.state.x_cmd[:2].astype(np.float32)
        start_v = self.state.v_cmd[:2].astype(np.float32)
        for bo in [0.02, 0.04]:
            hit_pos_2d = puck - dir_vec * (float(self.env_info["mallet"]["radius"]) + bo)
            cart = self._build_bezier(start_xy, start_v, hit_pos_2d, hit_vel_2d, T_plan)
            traj = self._optimize_joint_traj(cart)
            if traj:
                self._cnt_sac_ok += 1 
                return traj
        return []

    def _plan_fallback(self, observation):
        """ホームへ戻る簡易ベジエ。start≈goal ならYにナッジ。"""
        start_xy = self.state.x_cmd[:2].astype(np.float32)
        start_v = self.state.v_cmd[:2].astype(np.float32)
        goal_xy = self.agent_params['x_home'][:2].astype(np.float32).copy()

        # 退化回避ナッジ（±0.15m）
        if np.linalg.norm(goal_xy - start_xy) < 1e-3:
            puck_y = float(self.get_puck_pos(observation)[1])
            dy = 0.15 if puck_y >= 0.0 else -0.15
            y_lim = float(self.env_info['table']['width'] * 0.45)
            goal_xy[1] = float(np.clip(goal_xy[1] + dy, -y_lim, y_lim))

        cart = self._build_bezier(start_xy, start_v, goal_xy, np.zeros(2, dtype=np.float32), 1.0)

        # さらに退化する場合はXにも少し前進ナッジで再計画
        if cart is not None and hasattr(cart, "shape") and cart.shape[0] > 0:
            p = cart[:, 0:2]
            if np.linalg.norm(p[-1] - p[0]) < 1e-4:
                goal_xy2 = goal_xy.copy()
                goal_xy2[0] = min(self.table_length * 0.45, start_xy[0] + 0.05)
                cart = self._build_bezier(start_xy, start_v, goal_xy2, np.zeros(2, dtype=np.float32), 0.8)

        traj = self._optimize_joint_traj(cart)
        if LOG:
            if cart is None or not hasattr(cart, "size") or cart.size == 0:
                print("[MyAgent] PLAN=FB(empty:unopt)")
            else:
                p = cart[:, 0:2]
                print(f"[MyAgent] FB Δ={np.linalg.norm(p[-1]-p[0]):.6f}, len={cart.shape[0]}")
            print(f"[MyAgent] FB traj_len: {0 if not traj else len(traj)}")
        return traj
    
    def _build_policy_obs(self) -> np.ndarray:
        """
        SmashDecisionEnv と同じ 7 次元観測を組み立てる。
        [px, py, vx, vy, T, H, S]
        """
        s = np.asarray(self.state.estimated_state, dtype=np.float32)
        px = float(s[0]); py = float(s[1]); vx = float(s[2]); vy = float(s[3])

        # 直近の交差予測時刻（未設定なら 0）
        T = float(getattr(self.state, "predicted_time", 0.0))
        H = float(self.agent_params.get("max_prediction_time", 1.0))

        table_w = float(self.env_info["table"]["width"])
        table_l = float(self.env_info["table"]["length"])
        x_offset = float(self.env_info.get("table", {}).get("x_offset", 0.0))

        x_span = max(table_l * 0.5 - x_offset, 1e-6)  # stop_line→センターまで
        y_span = max(table_w * 0.5, 1e-6)
        v_scale = float(self.agent_params.get("puck_v_max", 3.0))

        # 位置 [-1,1]
        px01 = (px - x_offset) / x_span                # 0..1 目安
        px_n  = float(np.clip(2.0 * px01 - 1.0, -1.0, 1.0))
        py_n  = float(np.clip(py / y_span, -1.0, 1.0))

        # 速度 [-1,1]
        vx_n = float(np.clip(vx / v_scale, -1.0, 1.0))
        vy_n = float(np.clip(vy / v_scale, -1.0, 1.0))

        TH_n = float(2.0 * np.clip(T / max(H, 1e-6), 0.0, 1.0) - 1.0)                  
        H_n  = float(np.clip(H / 1.0, 0.0, 1.0))                         # だいたい 1.0 付近

        # ゲートの向き（符号）
        S_gate = (x_offset - px) * vx
        S_n = float(np.tanh(10.0 * S_gate))                              # [-1,1] に圧縮

        # 予測地平線・自陣距離・接近度
        return np.array([px_n, py_n, vx_n, vy_n, TH_n, H_n, S_n], dtype=np.float32)

    # ---------- メイン ----------
    def draw_action(self, observation):
        self._frame += 1
        if getattr(self, "new_start", False):
            self.state.reset()
            self.new_start = False

        # 観測更新
        self.state.update_observation(
            self.get_joint_pos(observation),
            self.get_joint_vel(observation),
            self.get_puck_pos(observation)
        )

        # 既に軌道があればそれを消化
        if len(self.state.trajectory_buffer) > 0:
            q, dq = self.state.trajectory_buffer[0]
            self.state.trajectory_buffer = self.state.trajectory_buffer[1:]
            self.state.x_cmd, self.state.v_cmd = self.state.update_ee_pos_vel(q, dq)
            return np.vstack([q, dq])

        # 1) まずSACプランを試す（学習結果を最優先で使う）
        if self.sac_model is not None and self.enable_sac:
            traj = self._try_sac_plan(observation)
            if traj:
                self.state.trajectory_buffer = traj
                self._cnt_sac += 1

        # 2) だめならフォールバック・ベジエ
        if len(self.state.trajectory_buffer) == 0:
            traj = self._plan_fallback(observation)
            if traj:
                self.state.trajectory_buffer = traj
                self._cnt_fb += 1

        # 3) それでも空ならBaselineに丸投げ（必ず動く）
        if len(self.state.trajectory_buffer) == 0:
            if LOG: print("[MyAgent] PLAN=BASELINE")
            self._cnt_base += 1
            return self.baseline.draw_action(observation)

        # 取り出して返す
        q, dq = self.state.trajectory_buffer[0]
        self.state.trajectory_buffer = self.state.trajectory_buffer[1:]
        self.state.x_cmd, self.state.v_cmd = self.state.update_ee_pos_vel(q, dq)

        self._frame += 1

        if LOG and (self._frame % self._log_every == 0):
            print(
                f"[MyAgent] summary(last{self._log_every}): "
                f"SAC_try={self._cnt_sac_try}, SAC_ko = {self._cnt_sac_ok}, "
                f"FB={self._cnt_fb}, BASE={self._cnt_base}"
                )
            self._cnt_sac_try = self._cnt_sac_ok = self._cnt_fb = self._cnt_base = 0

        return np.vstack([q, dq])
