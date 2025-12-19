from stable_baselines3 import PPO
import glob
import os
import time
import numpy as np
from patrol_env import env as env_f
from sb3_contrib import RecurrentPPO
import imageio


def evaluate(num_games: int = 100, render_mode=None, **env_kwargs):
    os.makedirs("gifs", exist_ok=True)

    env = env_f(render_mode=render_mode, **env_kwargs)
    print(env.unwrapped.action_space(env.possible_agents[0]))
    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        raise FileNotFoundError("Policy not found.")

    model = PPO.load(latest_policy)

    rewards = {agent: 0.0 for agent in env.possible_agents}
    patroller_win_count = 0
    intruder_success_count = 0

    for i in range(num_games):
        env.reset(seed=i)
        ep_rewards = {agent: 0.0 for agent in env.possible_agents}

        frames = []

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            ep_rewards[agent] += float(reward)
            if termination or truncation:
                env.step(None)
                continue

            act, _ = model.predict(obs, deterministic=True)
            env.step(act)

            if env.render_mode == "rgb_array":
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

        scen = env.unwrapped.scenario
        intruder_success_count += int(getattr(scen, "intruder_won", False))
        patroller_win_count += int(getattr(scen, "intruder_caught", False) and not getattr(scen, "intruder_won", False))

        for agent in env.possible_agents:
            rewards[agent] += ep_rewards[agent]

        if env.render_mode == "rgb_array" and len(frames) > 0:
            gif_path = os.path.join("gifs", f"game_{i + 1}.gif")
            imageio.mimsave(gif_path, frames, fps=40, loop=0)
            print(f"Saved GIF for game {i + 1} to {gif_path}")

    env.close()

    avg_reward_per_agent = {a: rewards[a] / num_games for a in rewards}
    avg_reward = float(np.mean(list(avg_reward_per_agent.values())))
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)

    patrollers = [a for a in avg_reward_per_agent if "patroller" in a]
    intruders = [a for a in avg_reward_per_agent if "intruder" in a]

    patroller_team_mean = float(np.mean([avg_reward_per_agent[a] for a in patrollers]))
    intruder_mean = float(np.mean([avg_reward_per_agent[a] for a in intruders]))

    print("Patroller team mean score:", patroller_team_mean)
    print("Intruder score:", intruder_mean)
    print("Patroller win rate:", patroller_win_count / num_games)
    print("Intruder win rate:", intruder_success_count / num_games)

    return avg_reward

def evaluate_optim(model, env, num_games: int = 10, render_mode=None, **env_kwargs):
    rewards = {agent: 0 for agent in env.possible_agents}

    for i in range(num_games):
        env.reset(seed=i)
        frames = []

        for _ in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            if env.render_mode == "rgb_array":
                frame = env.render()
                frames.append(frame)

            if termination or truncation:
                break
            else:
                act = model.predict(obs, deterministic=True)[0]

            env.step(act)

        for agent in env.agents:
            rewards[agent] += env.rewards[agent]

    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print(f"Avg reward across {num_games} games: {avg_reward}")
    return avg_reward
