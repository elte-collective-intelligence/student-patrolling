import pandas as pd
import matplotlib.pyplot as plt

RUN_ID = "20251220-132003"
LOG_PATH = f"patrolling/log/{RUN_ID}/progress.csv"

df = pd.read_csv(LOG_PATH)

steps = df["time/total_timesteps"]
mean_ep_len = df["roll100/ep_len_mean"]

plt.figure(figsize=(10, 6))
plt.plot(steps, mean_ep_len, linewidth=2)

plt.title("Mean Episode Length During Training")
plt.xlabel("Total Timesteps")
plt.ylabel("Episode Length")
plt.grid(alpha=0.4)

plt.savefig("images/mean_ep_length.png", dpi=300)
plt.show()
