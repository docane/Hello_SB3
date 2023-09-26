import gymnasium as gym
from typing import Callable
from stable_baselines3.common.utils import set_random_seed


# 벡터화된 환경을 함수로 생성
def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func
