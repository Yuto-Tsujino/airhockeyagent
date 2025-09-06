import numpy as np
from stable_baselines3 import SAC

from air_hockey_challenge.framework import AgentBase
from baseline.baseline_agent.baseline_agent import BaselineAgent
from baseline.baseline_agent.system_state import SystemState

class HybridAgent(AgentBase):
    def __init__(self, env_info, agent_id=1, sac_model_path=None, **kwargs):
        super().__init__(env_info, agent_id, **kwargs)

        # Baseline を内部に持つ
        self.baseline = BaselineAgent(env_info, agent_id=agent_id, **kwargs)

        # SAC モデルをロード（未学習なら None）
        if sac_model_path is not None:
            self.sac_model = SAC.load(sac_model_path)
        else:
            self.sac_model = None

        self.env_info = env_info
        self.agent_id = agent_id

    def reset(self):
        self.baseline.reset()

    def draw_action(self, obs):
        """
        Baseline の戦術切り替えに従う。
        攻撃戦術なら SAC を呼ぶ。
        それ以外は Baseline の行動を返す。
        """
        # Baseline の内部状態を更新
        self.baseline.state.update_observation(
            self.baseline.get_joint_pos(obs),
            self.baseline.get_joint_vel(obs),
            self.baseline.get_puck_pos(obs),
        )

        if len(self.baseline.state.trajectory_buffer) > 0:
            q, dq = self.baseline.state.trajectory_buffer[0]
            self.baseline.state.trajectory_buffer = self.base.state.trajectory_buffer[1:]
            self.baseline.state.x_cmd, self.baseline.state.v_cmd = \
                self.baseline.state.update_ee_pos_vel(q, dq)
            return np.vstack([q, dq])

        self.baseline.tactics_processor[self.baseline.state.tactic_current.value].update_tactic()

        # 現在の戦術
        tactic_id = self.baseline.state.tactic_current.value
        tactic_name = self.baseline.tactics_processor[tactic_id].__class__.__name__

        if tactic_name in ["Smash", "Repel"]:  # 攻撃タクティクス
            if self.sac_model is not None:
                # 観測を SAC に渡す
                obs_vec = self._build_obs_for_sac()
                action, _ = self.sac_model.predict(obs_vec, deterministic=True)

                # (dY, alpha, beta) → Baseline の trajectory generator に適用する
                return self._apply_attack_action(action, obs)

            else:
                # SAC が未学習の場合はランダム行動
                return self.baseline.draw_action(obs)

        # それ以外の戦術（defend, prepare など）は Baseline の行動をそのまま返す
        return self.baseline.draw_action(obs)

    def _build_obs_for_sac(self):
        """
        SmashDecisionEnv 相当の観測ベクトルを構築
        （ここは簡略化、詳細は学習環境側に合わせてチューニング）
        """
        puck = self.baseline.state.puck
        px, py = puck[0], puck[1]
        vx, vy = puck[2], puck[3]
        # 正規化などは SmashDecisionEnv と同じにしておくのが望ましい
        obs = np.array([px, py, vx, vy], dtype=np.float32)
        return obs

    def _apply_attack_action(self, action, obs):
        """
        SAC が出した (dY, alpha, beta) を使って攻撃動作を生成
        → Baseline の軌道生成に渡す
        """
        # ここは train_smash_sac.py の学習時の仕様に合わせる必要がある
        # 仮実装としては Baseline の Smash 動作を模倣しつつ action を反映させる
        return self.baseline.draw_action(obs)
