import optuna
import supersuit as ss
from stable_baselines3 import PPO
import time
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from train import train
import glob
from eval import evaluate_optim
import os
from patrol_env import env as env_f

def optimize_train(trial):
    """
    Objective function for Optuna to optimize PPO hyperparameters.
    """
    hyperparams = {
        "batch_size": trial.suggest_categorical("batch_size", [256, 512]),
        "n_steps": trial.suggest_int("n_steps", 2048, 8192, step=1024),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
        "gamma": trial.suggest_float("gamma", 0.8, 0.99),
        "net_arch": trial.suggest_categorical(
            "net_arch", ["[64, 32]", "[128, 64, 32]", "[256, 128, 64]"]
        ),
    }

    hyperparams["net_arch"] = eval(hyperparams["net_arch"])
    env = env_f(render_mode="rgb_array")

    train(
        env_fn="patrolEnv",
        steps=20000, 
        seed=42,
        hyperparams=hyperparams,
    )

    latest_policy_path = max(
        glob.glob(f"{env.unwrapped.metadata.get('name')}*.zip"),
        key=os.path.getctime,
    )
    model = PPO.load(latest_policy_path)

    avg_reward = evaluate_optim(
        model=model,
        env=env,
        num_games=10,
        render_mode=None,
    )

    return avg_reward


study = optuna.create_study(direction="maximize")
study.optimize(optimize_train, n_trials=20)

print("Best hyperparameters:")
print(study.best_params)