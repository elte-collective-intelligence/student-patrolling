import pandas as pd
import matplotlib.pyplot as plt

RUN_ID = "20251220-132003"
LOG_PATH = f"patrolling/log/{RUN_ID}/progress.csv"

df = pd.read_csv(LOG_PATH)

# X and Y from SB3 logs
steps = df["time/total_timesteps"]
mean_reward = df["roll100/reward_mean"]

plt.figure(figsize=(10, 6))
plt.plot(steps, mean_reward, linewidth=2)

plt.title("Mean Reward During Training")
plt.xlabel("Total Timesteps")
plt.ylabel("Mean Reward (rolling 100)")
plt.grid(alpha=0.4)

plt.savefig("images/mean_reward.png", dpi=300)
plt.show()
