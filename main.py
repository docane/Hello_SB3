from exmaples import *

# 행동이 이산적인 CartPole 환경 사용
env_id = 'CartPole-v1'
learn_dqn(env_id)
# learn_dqn_multi_agent(env_id, num_envs=6)
# learn_a2c_multi_agent(env_id, num_envs=6)
# learn_ppo_callback(env_id)
# learn_ppo_eval_callback(env_id)
# learn_ppo_linear_schedule(env_id)
# learn_ppo_policy_kwargs(env_id)
# learn_ppo_checkpoint_callback(env_id)

# 행동이 연속적인 MountainCar 환경 사용
# env_id = 'MountainCarContinuous-v0'
# learn_sac(env_id)

# 상태가 이미지인 Breakout 환경 사용
# env_id = 'BreakoutNoFrameskip-v4'
# learn_ppo_cnn(env_id)
