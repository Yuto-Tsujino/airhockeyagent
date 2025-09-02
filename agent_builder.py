from .MyAgent import MyAgent

def build_agent(env_info, **kwargs):
    return MyAgent(env_info, agent_id=1, **kwargs)





# # C:\Airhockeychallenge2025\air_hockey_agent\agent_builder.py

# # 既存のベースラインエージェントのコードを直接インポートします。
# # このパスは C:\Airhockeychallenge2025\baseline\baseline_agent\baseline_agent.py を指します。
# import numpy as np
# import torch
# from baseline.baseline_agent.baseline_agent import BaselineAgent
# from baseline.baseline_agent.system_state import TACTICS
# from baseline.baseline_agent.tactics import Smash
# from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper

# class SimpleSmashModel(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 例：入力層、中間層、出力層を持つ簡単なネットワーク
#         self.network = torch.nn.Sequential(
#             torch.nn.Linear(6, 32), # (パック位置_xyz, パック速度_xyz)の6次元を入力と仮定
#             torch.nn.ReLU(),
#             torch.nn.Linear(32, 2)  # 攻撃目標のxy座標の2次元を出力
#         )
    
#     def forward(self, x):
#         return self.network(x)
# # torch.serialization.add_safe_globals([
# #     SimpleSmashModel, 
# #     torch.nn.Sequential,
# #     torch.nn.Linear,
# #     torch.nn.ReLU])

# class MyLearningAgent(BaselineAgent):
#     """
#     BaselineAgentを継承し、一部の振る舞いを変更する。
#     """
#     def __init__(self, env_info, **kwargs):
#         # まず、親クラス(BaselineAgent)の初期化を呼び出す
#         super().__init__(env_info, **kwargs)

#         self.smash_model = SimpleSmashModel()

#         # Smash戦術だけを、AIを搭載した独自の戦術に置き換える
#         self.tactics_processor[5] = MySmashTactic(
#             self.env_info, self.agent_params, self.state, 
#             self.traj_generator, self.smash_model)
        
#         self._add_save_attr(
#             smash_model='torch'
#         )

#     def fit(self, dataset, **kwargs):
#         print("Agent.fit() called, but not implemented yet.")
#         pass

# class MySmashTactic(Smash):
#     """
#     オリジナルのSmash戦術を継承し、applyメソッドだけを変更する
#     """
#     def __init__(self, env_info, agent_params, state, trajectory_generator, model):
#         super().__init__(env_info, agent_params, state, trajectory_generator)
#         self.model = model
#         self.model.eval() 
#         print("MySmashTactic has been correctly initialized!")

#         def apply(self):
#             print("★★★ My AI is deciding where to smash! ★★★")
#             self.state.tactic_current = TACTICS.READY
#             self.state.tactic_finish = True


# def build_agent(env_info, **kwargs):
#     """
#     MyLearningAgentを返す
#     """
#     return MyLearningAgent(env_info, **kwargs)

# # if __name__ == '__main__':
# #     env = AirHockeyChallengeWrapper("hit")

# #     print("/************************ SAVING AGENT ************************/")
# #     agent_to_save = build_agent(env.env_info)
# #     print("Model parameter BEFORE save:", list(agent_to_save.smash_model.parameters())[0].data)
# #     agent_to_save.save("my_agent.msh", full_save=True)
# #     print("Agent saved to my_agent.msh")
# #     print("\n/************************ LOADING AGENT ************************/")
# #     agent_loaded = MyLearningAgent.load("my_agent.msh")
# #     print("Model parameter AFTER load: ", list(agent_loaded.smash_model.parameters())[0].data)

# #     print("\nSave and Load successful!")
