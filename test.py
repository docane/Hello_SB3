import gymnasium as gym
from stable_baselines3 import DQN, A2C, PPO, SAC

env_id = 'CartPole-v1'
# env_id = 'MountainCarContinuous-v0'
# env_id = 'BreakoutNoFrameskip-v4'

env = gym.make(env_id, render_mode='human')

# CartPole 환경 사용
model = DQN.load('./models/dqn_cartpole-v1', env)
# model = DQN.load('./models/dqn_multi_agent_cartpole-v1', env)
# model = A2C.load('./models/a2c_multi_agent_cartpole-v1', env)
# model = PPO.load('./models/ppo_callback_cartpole-v1', env)
# model = PPO.load('./models/ppo_eval_callback_cartpole-v1', env)
# model = PPO.load('./models/ppo_linear_learning_rate_cartpole-v1_1', env)
# model = PPO.load('./models/ppo_linear_learning_rate_cartpole-v1_2', env)
# model = PPO.load('./models/ppo_policy_kwargs_cartpole-v1', env)
# model = PPO.load('./models/ppo_checkpoint_callback_cartpole-v1', env)

# MountainCarContinuous 환경 사용
# model = SAC.load('./models/sac_MountainCarContinuous-v0')

# BreakoutNoFrameskip 환경 사용
# model = PPO.load('./models/ppo_cnn_BreakoutNoFrameskip-v4')

state, info = env.reset()
terminated, truncated = False
while not (terminated or truncated):
    action, _states = model.predict(state, deterministic=True)
    state, reward, terminated, truncated, info = env.step(action)
    env.render()
