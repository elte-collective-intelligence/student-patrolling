import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("reward_log.csv")

steps = data["Step"].to_numpy() 
values = data["Value"].to_numpy() 

plt.figure(figsize=(10, 6))
plt.plot(steps, values, label="Reward per Episode", linewidth=2, color='blue')

plt.title("Reward Progression During Training", fontsize=16)
plt.xlabel("Training Steps", fontsize=14)
plt.ylabel("Reward", fontsize=14)

plt.grid(alpha=0.4)
plt.legend(fontsize=12)

plt.savefig("images/reward_progression.png", dpi=300)
plt.show()
