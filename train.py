import time
import numpy as np
import supersuit as ss

from patrol_env import parallel_env
from sb3_contrib import RecurrentPPO

from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from supersuit.vector.constructors import MakeCPUAsyncConstructor
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper


# -----------------------------
# Logging callback (optional)
# -----------------------------
class RewardLoggingCallback(BaseCallback):
    """
    Logs episode reward to TensorBoard if env puts episode info in info["episode"].
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos:
            for info in infos:
                ep = info.get("episode", None)
                if ep is not None and "r" in ep:
                    self.episode_rewards.append(ep["r"])
                    self.logger.record("reward/episode", ep["r"])
        return True


# -----------------------------
# Env builder for SB3
# -----------------------------
def sb3_compatible_env(seed: int, env_kwargs: dict):
    """
    Builds a PettingZoo parallel env, pads obs, converts to VecEnv for SB3.
    """
    e = parallel_env(**env_kwargs)
    e.reset(seed=seed)

    e = ss.multiagent_wrappers.pad_observations_v0(e)
    e = ss.pettingzoo_env_to_vec_env_v1(e)
    return e


def make_vec_env(seed: int, env_kwargs: dict, num_vec_envs: int = 8, num_cpus: int = 8):
    """
    Async CPU VecEnv for faster training.
    """
    tmp = sb3_compatible_env(seed=seed, env_kwargs=env_kwargs)
    obs_space = tmp.observation_space
    act_space = tmp.action_space
    tmp.close()

    def _factory(i):
        return lambda: sb3_compatible_env(seed=seed + i, env_kwargs=env_kwargs)

    env_fns = [_factory(i) for i in range(num_vec_envs)]
    cpu_async_vec = MakeCPUAsyncConstructor(num_cpus)(env_fns, obs_space, act_space)

    venv = SB3VecEnvWrapper(cpu_async_vec)
    venv = VecMonitor(venv, filename="./logs/")
    return venv

def evaluate_win_rate(model, env_kwargs, seeds, episodes_per_seed=1):
    patroller_wins = 0
    intruder_wins = 0
    timeouts = 0
    total = 0

    for seed in seeds:
        for _ in range(episodes_per_seed):
            env = make_eval_env(seed, env_kwargs)

            obs = env.reset()  # shape: (1, obs_dim)
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)

            done = False
            outcome = "timeout"

            while not done:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )

                obs, rewards, dones, infos = env.step(action)
                done = bool(dones[0])
                episode_starts = dones

                if done and infos:
                    ep = infos[0].get("episode", None)
                    if ep is not None:
                        outcome = ep.get("outcome", "timeout")

            env.close()
            total += 1

            if outcome == "patrollers_win":
                patroller_wins += 1
            elif outcome == "intruders_win":
                intruder_wins += 1
            else:
                timeouts += 1

    return (
        patroller_wins / total,
        intruder_wins / total,
        timeouts / total,
        total,
    )





def make_eval_env(seed: int, env_kwargs: dict):
    """
    Evaluation env MUST match SB3 VecEnv format.
    Single environment, same wrappers as training.
    """
    env = sb3_compatible_env(seed=seed, env_kwargs=env_kwargs)
    env = VecMonitor(env)
    return env




def calibrate_to_50_percent(
    model: RecurrentPPO,
    base_env_kwargs: dict,
    difficulty_values,
    calib_seeds,
    episodes_per_seed: int = 1,
    target: float = 0.5
):
    """
    Sweeps one difficulty knob (e.g., num_obstacles) to get patroller win-rate close to 50%.
    Returns: (best_value, results_dict)
    """
    results = {}
    best_val = None
    best_gap = float("inf")

    for val in difficulty_values:
        env_kwargs = dict(base_env_kwargs)
        env_kwargs["num_obstacles"] = val  # <-- difficulty knob (edit if you choose another)

        pw, iw, tr, n = evaluate_win_rate(
            model=model,
            env_kwargs=env_kwargs,
            seeds=calib_seeds,
            episodes_per_seed=episodes_per_seed
        )

        results[val] = {"patroller_wr": pw, "intruder_wr": iw, "timeout_r": tr, "episodes": n}
        gap = abs(pw - target)

        print(f"[CALIB] num_obstacles={val} | patroller_wr={pw:.3f} (gap {gap:.3f}) | n={n}")

        if gap < best_gap:
            best_gap = gap
            best_val = val

    return best_val, results


# -----------------------------
# Training (Task 4.1 baseline)
# -----------------------------
def train(steps: int = 100_000, seed: int = 0, hyperparams=None, **env_kwargs):
    if hyperparams is None:
        hyperparams = {}

    tmp_path = "./patrolling/log/"
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    # training env
    env = make_vec_env(seed=seed, env_kwargs=env_kwargs, num_vec_envs=8, num_cpus=8)

    model = RecurrentPPO(
        "MlpLstmPolicy",               # <-- RL2-style baseline (recurrent policy)
        env,
        verbose=1,
        batch_size=hyperparams.get("batch_size", 512),
        n_steps=hyperparams.get("n_steps", 128),
        tensorboard_log=tmp_path,
        ent_coef=hyperparams.get("ent_coef", 0.01),
        clip_range=hyperparams.get("clip_range", 0.2),
        learning_rate=hyperparams.get("learning_rate", 3e-4),
        vf_coef=hyperparams.get("vf_coef", 0.5),
        n_epochs=hyperparams.get("n_epochs", 10),
        gae_lambda=hyperparams.get("gae_lambda", 0.95),
        max_grad_norm=hyperparams.get("max_grad_norm", 0.5),
        gamma=hyperparams.get("gamma", 0.99),
        normalize_advantage=True,
        policy_kwargs=dict(
            net_arch=dict(
                pi=hyperparams.get("net_arch_pi", [256, 128]),
                vf=hyperparams.get("net_arch_vf", [256, 128]),
            )
        ),
    )

    model.set_logger(new_logger)

    reward_logging_callback = RewardLoggingCallback()

    print(f"Starting training on {env_kwargs.get('mode', 'train')} | steps={steps} | seed={seed}")
    model.learn(
        total_timesteps=steps,
        tb_log_name="RecurrentPPO_RL2_Baseline",
        callback=[reward_logging_callback],
        log_interval=10
    )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    save_name = f"recurrentppo_patrolling_{stamp}"
    model.save(save_name)
    print(f"Saved model: {save_name}")

    env.close()
    return model


# -----------------------------
# Task 4.2: calibrate + held-out meta-eval
# -----------------------------
def run_task4_2(model: RecurrentPPO, base_env_kwargs: dict):
    # 1) calibration seeds and held-out seeds
    calib_seeds = list(range(0, 20))
    heldout_seeds = list(range(1000, 1020))

    # 2) choose difficulty sweep (example: num_obstacles)
    difficulty_values = [2, 3, 5, 7, 9, 12]

    # 3) calibrate to ~50% patroller win-rate
    best_val, calib_results = calibrate_to_50_percent(
        model=model,
        base_env_kwargs=base_env_kwargs,
        difficulty_values=difficulty_values,
        calib_seeds=calib_seeds,
        episodes_per_seed=1,
        target=0.5
    )

    print(f"\n[CALIB DONE] Best num_obstacles={best_val} (closest to 50% patroller win-rate)\n")

    # 4) meta-evaluate on held-out maps (no more tuning!)
    eval_env_kwargs = dict(base_env_kwargs)
    eval_env_kwargs["num_obstacles"] = best_val

    pw, iw, tr, n = evaluate_win_rate(
        model=model,
        env_kwargs=eval_env_kwargs,
        seeds=heldout_seeds,
        episodes_per_seed=1
    )

    print("[HELD-OUT META-EVAL]")
    print(f"num_obstacles={best_val}")
    print(f"patroller_win_rate={pw:.3f} | intruder_win_rate={iw:.3f} | timeout_rate={tr:.3f} | n={n}")

    return best_val, calib_results, {"patroller_wr": pw, "intruder_wr": iw, "timeout_r": tr, "episodes": n}


if __name__ == "__main__":
    # Example usage:
    # 1) Train
    model = train(
        steps=200_000,
        seed=0,
        num_intruders=1,
        num_patrollers=3,
        num_obstacles=5,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        mode="train",
    )

    # 2) Run Task 4.2
    base_kwargs = dict(
        num_intruders=1,
        num_patrollers=3,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        mode="train",
    )
    run_task4_2(model, base_kwargs)
