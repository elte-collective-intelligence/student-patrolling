import pandas as pd
import matplotlib.pyplot as plt

RUN_ID = "20251220-132003"
MONITOR_PATH = f"patrolling/log/{RUN_ID}/monitor.csv"

# SB3 monitor.csv has a comment header line starting with "#"
df = pd.read_csv(MONITOR_PATH, skiprows=1)

# Raw episode reward (return)
ep_reward = df["r"].to_numpy()

# Episode index as x-axis (or you could use cumulative timesteps if you want)
episodes = range(1, len(ep_reward) + 1)

plt.figure(figsize=(10, 6))
plt.plot(episodes, ep_reward, linewidth=1)

plt.title("Reward per Episode (Raw)", fontsize=16)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Episode Reward", fontsize=14)

plt.grid(alpha=0.4)
plt.savefig("images/reward_progression.png", dpi=300)
plt.show()
