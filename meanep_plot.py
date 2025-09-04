import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("mean_ep.csv")

steps = data["Step"].to_numpy()
mean_ep_length = data["Value"].to_numpy() 

plt.figure(figsize=(10, 6))
plt.plot(steps, mean_ep_length, label="Mean Episode Length", linewidth=2, color='blue')

plt.title("Average Episode Length During Evaluation", fontsize=16)
plt.xlabel("Evaluation Steps", fontsize=14)
plt.ylabel("Mean Episode Length", fontsize=14)

plt.grid(alpha=0.4)
plt.legend(fontsize=12)

plt.savefig("images/mean_ep_length.png", dpi=300)
plt.show()
