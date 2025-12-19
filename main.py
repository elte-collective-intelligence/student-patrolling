from pettingzoo.mpe._mpe_utils.simple_env import make_env
import glob
import os
import time
import numpy as np
import time
from eval import evaluate
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy

from train import train
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN
from stable_baselines3.dqn import CnnPolicy, MlpPolicy
from pettingzoo.utils.conversions import (
    aec_to_parallel_wrapper,
    parallel_to_aec_wrapper,
    turn_based_aec_to_parallel_wrapper,
)
from pettingzoo.utils.wrappers import BaseWrapper
import warnings
from pettingzoo.test.api_test import missing_attr_warning

def main():
    env_kwargs = dict(
        max_cycles=120,
        continuous_actions=False,
        num_intruders=1, 
        num_patrollers=4, 
        num_obstacles=5
    )
    
    env_fn = "patrolEnv"
    train(env_fn, steps=int(5e4), seed=16, render_mode=None, hyperparams=None ,**env_kwargs)
    evaluate(env_fn, num_games=5, render_mode="human", **env_kwargs)

if __name__ == "__main__":
    main()