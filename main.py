print(">>> RUNNING MODIFIEDxxxxx MAIN.PY <<<")

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

from train import train, run_task4_2
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
        num_intruders=1,
        num_patrollers=3,
        num_obstacles=5,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        mode="train",
    )

    # 1) Train meta-learning baseline
    model = train(
        steps=100_000,
        seed=16,
        **env_kwargs
    )

    # 2) Run Task 4.2 calibration + held-out evaluation
    base_kwargs = dict(
        num_intruders=1,
        num_patrollers=3,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        mode="train",
    )

    run_task4_2(model, base_kwargs)

if __name__ == "__main__":
    main()