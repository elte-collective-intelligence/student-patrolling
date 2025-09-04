from stable_baselines3 import PPO
import glob
import os
import time
import numpy as np
from patrol_env import env as env_f
from sb3_contrib import RecurrentPPO
import imageio


def evaluate(env_fn, num_games: int = 100, render_mode=None, **env_kwargs):
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
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        frames = []

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            if env.render_mode == "rgb_array":
                frame = env.render()
                frames.append(frame)

            for agent in env.agents:
                rewards[agent] += env.rewards[agent]

            if termination or truncation:
                break
            else:
                act = model.predict(obs, deterministic=True)[0]

            env.step(act)
        if env.render_mode == "human":
            env.render()

        # Save the frames as a GIF
        """gif_path = os.path.join("gifs", f"game_{i + 1}.gif")
        imageio.mimsave(gif_path, frames, fps=40, loop=0)
        print(f"Saved GIF for game {i + 1} to {gif_path}")"""

    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)

    return avg_reward

def evaluate_optim(model, env, num_games: int = 10, render_mode=None, **env_kwargs):
    rewards = {agent: 0 for agent in env.possible_agents}

    for i in range(num_games):
        env.reset(seed=i)
        frames = []

        for agent in env.agent_iter():
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
