import supersuit as ss
from patrol_env import parallel_env
from stable_baselines3 import PPO
import time
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback
from stable_baselines3.common.vec_env import VecMonitor

class RewardLoggingCallback(BaseCallback):
    """
    Custom callback to log rewards to TensorBoard.
    """
    def __init__(self, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info.keys():
                    reward = info["episode"]["r"]
                    self.episode_rewards.append(reward)
                    self.logger.record("reward/episode", reward)
        return True

def train(env_fn, steps: int = 100, seed=0, hyperparams=None, **env_kwargs):
    """
    Train the PPO model with customizable hyperparameters.
    Args:
        env_fn: Environment creation function.
        steps: Total timesteps for training.
        seed: Random seed.
        hyperparams: Dictionary of hyperparameters.
        env_kwargs: Additional arguments for environment creation.
    """
    if hyperparams is None:
        hyperparams = {}

    env = parallel_env(**env_kwargs)
    env.reset(seed=seed)

    tmp_path = "./patrolling/log/"
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    print(f"Starting training on {str(env.metadata['name'])}.")
    env = ss.multiagent_wrappers.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 20, num_cpus=0, base_class="stable_baselines3")
    env = VecMonitor(env, filename="./logs/")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        batch_size=hyperparams.get("batch_size", 512),
        n_steps=hyperparams.get("n_steps", 128),
        tensorboard_log=tmp_path,
        ent_coef=hyperparams.get("ent_coef", 0.09964),
        clip_range=0.2,
        learning_rate=hyperparams.get("learning_rate", 1.08e-5),
        vf_coef=hyperparams.get("vf_coef", 0.7),
        n_epochs=hyperparams.get("n_epochs", 15),
        gae_lambda=hyperparams.get("gae_lambda", 0.9),
        max_grad_norm=hyperparams.get("max_grad_norm", 0.3),
        gamma=hyperparams.get("gamma", 0.803743),
        normalize_advantage=True,
        policy_kwargs=dict(
            net_arch=dict(
                pi=hyperparams.get("net_arch", [256, 128, 32]),
                vf=hyperparams.get("net_arch", [256, 128, 32]),
            )
        ),
    )

    model.set_logger(new_logger)

    eval_callback = EvalCallback(
        env,
        best_model_save_path="./patrolling/log/best_model/",
        log_path="./patrolling/log/eval_logs/",
        eval_freq=128,
        deterministic=True,
        render=False,
    )

    reward_logging_callback = RewardLoggingCallback()

    model.learn(
        total_timesteps=steps,
        tb_log_name="MLP_Policy",
        callback=[eval_callback, reward_logging_callback],
        log_interval=1
    )

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")
    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")
    env.close()