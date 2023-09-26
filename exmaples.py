import os
import torch as th
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, A2C, PPO, SAC
from stable_baselines3.common import results_plotter
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, \
    StopTrainingOnRewardThreshold, CallbackList
from stable_baselines3.common.results_plotter import plot_results
from utils import make_env, linear_schedule
from callback import SaveOnBestTrainingRewardCallback


def learn_dqn(env_id, total_timesteps=200000):
    model = DQN('MlpPolicy', env_id, verbose=1)
    model.learn(total_timesteps, progress_bar=True)
    model.save(f'./models/dqn_{env_id}')


def learn_dqn_multi_agent(env_id, num_envs=1, total_timesteps=200000):
    vec_env = make_vec_env(env_id, n_envs=num_envs, vec_env_cls=DummyVecEnv)
    model = DQN('MlpPolicy', vec_env, verbose=1)
    model.learn(total_timesteps)
    model.save(f'./models/dqn_multi_agent_{env_id}')


# 벡터화된 환경을 SubprocVecEnv 함수를 이용하여 생성하고 학습할 때 사용
def learn_a2c_multi_agent(env_id, num_envs=1, total_timesteps=200000):
    if __name__ == "__main__":
        envs = [make_env(env_id, i) for i in range(num_envs)]
        vec_env = SubprocVecEnv(envs)

        model = A2C('MlpPolicy', vec_env, verbose=1)
        model.learn(total_timesteps)
        model.save(f'./models/a2c_multi_agent_{env_id}')


def learn_ppo_callback(env_id, total_timesteps=100000):
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(env_id)
    env = Monitor(env, log_dir)

    model = PPO('MlpPolicy', env, verbose=1)
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps, callback=callback)
    model.save(f'./models/ppo_callback_{env_id}')

    plot_results([log_dir], total_timesteps, results_plotter.X_TIMESTEPS, "PPO CartPole")
    plt.show()


def learn_ppo_eval_callback(env_id, n_training_envs=1, n_eval_envs=5, total_timesteps=50000):
    eval_log_dir = "./eval_logs/"
    os.makedirs(eval_log_dir, exist_ok=True)

    train_env = make_vec_env(env_id, n_envs=n_training_envs, seed=0)
    eval_env = make_vec_env(env_id, n_envs=n_eval_envs, seed=0)

    eval_callback = EvalCallback(eval_env, best_model_save_path=eval_log_dir,
                                 log_path=eval_log_dir, eval_freq=max(500 // n_training_envs, 1),
                                 n_eval_episodes=5, deterministic=True,
                                 render=False)

    model = PPO("MlpPolicy", train_env)
    model.learn(total_timesteps, callback=eval_callback)
    model.save(f'./models/ppo_eval_callback_{env_id}')


def learn_ppo_linear_schedule(env_id):
    model = PPO('MlpPolicy', env_id, learning_rate=linear_schedule(0.001), verbose=1)
    model.learn(total_timesteps=20000)
    model.save(f'./models/ppo_linear_learning_rate_{env_id}_1')
    model.learn(total_timesteps=10000, reset_num_timesteps=True)
    model.save(f'./models/ppo_linear_learning_rate_{env_id}_2')


def learn_sac(env_id, total_timesteps=20000):
    model = SAC('MlpPolicy', env_id, verbose=1)
    model.learn(total_timesteps)
    model.save(f'./models/sac_{env_id}')


def learn_ppo_policy_kwargs(env_id, total_timesteps=100000, policy_kwargs=None):
    if policy_kwargs is None:
        policy_kwargs = [32, 32]
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=dict(pi=policy_kwargs, vf=policy_kwargs))
    model = PPO('MlpPolicy', env_id, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps)
    model.save(f'./models/ppo_policy_kwargs_{env_id}')


def learn_ppo_checkpoint_callback(env_id, total_timesteps=100000):
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/')
    eval_env = gym.make(env_id)
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 best_model_save_path='./logs/',
                                 eval_freq=500,
                                 deterministic=True,
                                 render=False)

    callback = CallbackList([checkpoint_callback, eval_callback])
    model = PPO('MlpPolicy', env_id, verbose=1, tensorboard_log='./tensorboard_logs/ppo_cartpole/')
    model.learn(total_timesteps, callback=callback, progress_bar=True)
    model.save(f'./models/ppo_checkpoint_callback_{env_id}')


def learn_ppo_cnn(env_id, total_timesteps=1000):
    model = PPO('CnnPolicy', env_id, verbose=1)
    model.learn(total_timesteps)
    model.save(f'ppo_cnn_{env_id}')