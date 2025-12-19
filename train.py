import supersuit as ss
from patrol_env import parallel_env
from stable_baselines3 import PPO
import time
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from supersuit.vector.constructors import MakeCPUAsyncConstructor
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper
import os
from collections import deque
import numpy as np

class RewardLoggingCallback(BaseCallback):
    """
    Custom callback to log rewards to TensorBoard.
    """

    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size

        self.ep_reward = deque(maxlen=window_size)
        self.ep_len = deque(maxlen=window_size)

        self.patroller_win = deque(maxlen=window_size)
        self.intruder_success = deque(maxlen=window_size)
        self.intruder_caught = deque(maxlen=window_size)

        self.time_to_capture = deque(maxlen=window_size)
        self.time_to_intruder_success = deque(maxlen=window_size)

    @staticmethod
    def _mean(dq: deque):
        return float(np.mean(dq)) if len(dq) > 0 else None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        for info in infos:
            if "episode" in info:
                ep = info["episode"]
                if "r" in ep:
                    r = float(ep["r"])
                    self.ep_reward.append(r)
                    self.logger.record("episode/reward", r)
                if "l" in ep:
                    l = float(ep["l"])
                    self.ep_len.append(l)
                    self.logger.record("episode/length", l)

                if "patroller_win" in info:
                    v = float(info["patroller_win"])
                    self.patroller_win.append(v)
                    self.logger.record("episode/patroller_win", v)

                if "intruder_success" in info:
                    v = float(info["intruder_success"])
                    self.intruder_success.append(v)
                    self.logger.record("episode/intruder_success", v)

                if "intruder_caught" in info:
                    v = float(info["intruder_caught"])
                    self.intruder_caught.append(v)
                    self.logger.record("episode/intruder_caught", v)

                if info.get("time_to_capture", None) is not None:
                    v = float(info["time_to_capture"])
                    self.time_to_capture.append(v)
                    self.logger.record("episode/time_to_capture", v)

                if info.get("time_to_intruder_success", None) is not None:
                    v = float(info["time_to_intruder_success"])
                    self.time_to_intruder_success.append(v)
                    self.logger.record("episode/time_to_intruder_success", v)

        m = self._mean(self.ep_reward)
        if m is not None:
            self.logger.record(f"roll{self.window_size}/reward_mean", m)

        m = self._mean(self.ep_len)
        if m is not None:
            self.logger.record(f"roll{self.window_size}/ep_len_mean", m)

        m = self._mean(self.patroller_win)
        if m is not None:
            self.logger.record(f"roll{self.window_size}/win_rate", m)

        m = self._mean(self.intruder_success)
        if m is not None:
            self.logger.record(f"roll{self.window_size}/intruder_success_rate", m)

        m = self._mean(self.intruder_caught)
        if m is not None:
            self.logger.record(f"roll{self.window_size}/intruder_caught_rate", m)

        m = self._mean(self.time_to_capture)
        if m is not None:
            self.logger.record(f"roll{self.window_size}/time_to_capture_mean", m)

        m = self._mean(self.time_to_intruder_success)
        if m is not None:
            self.logger.record(f"roll{self.window_size}/time_to_intruder_success_mean", m)

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

    run_id = time.strftime("%Y%m%d-%H%M%S")
    tmp_path = f"./patrolling/log/{run_id}/"
    os.makedirs(tmp_path, exist_ok=True)
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    def sb3_compatible(seed_: int, env_kwargs_: dict):
        e = parallel_env(**env_kwargs_)
        e.reset(seed=seed_)

        e = ss.multiagent_wrappers.pad_observations_v0(e)
        e = ss.pettingzoo_env_to_vec_env_v1(e)
        return e

    num_vec_envs = 8
    num_cpus = 8

    tmp_env = sb3_compatible(seed_=seed, env_kwargs_=env_kwargs)
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    tmp_env.close()

    def _factory(i):
        return lambda: sb3_compatible(seed_=(seed + i), env_kwargs_=env_kwargs)

    env_fns = [_factory(i) for i in range(num_vec_envs)]

    cpu_async_vec = MakeCPUAsyncConstructor(num_cpus)(env_fns, obs_space, act_space)

    env = SB3VecEnvWrapper(cpu_async_vec)
    env = VecMonitor(env, filename=os.path.join(tmp_path, "monitor.csv"))

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        batch_size=hyperparams.get("batch_size", 512),
        n_steps=hyperparams.get("n_steps", 128),
        tensorboard_log=tmp_path,
        ent_coef=hyperparams.get("ent_coef", 0.09964),
        clip_range=hyperparams.get("clip_range", 0.2),
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

    eval_env = sb3_compatible(seed_=(seed + 10_000), env_kwargs_=env_kwargs)
    eval_env = SB3VecEnvWrapper(eval_env)
    eval_env = VecMonitor(eval_env, filename=os.path.join(tmp_path, "eval_monitor.csv"))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(tmp_path, "best_model"),
        log_path=os.path.join(tmp_path, "eval_logs"),
        eval_freq=int(hyperparams.get("eval_freq", 100)),
        n_eval_episodes=int(hyperparams.get("n_eval_episodes", 20)),
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