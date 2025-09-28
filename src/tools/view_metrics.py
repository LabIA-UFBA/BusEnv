import pandas as pd
import matplotlib.pyplot as plt

# Load metrics file
df = pd.read_csv("env_metrics.csv")

# Group by episode to avoid multiple entries per episode
df_grouped = df.groupby("episode").mean().reset_index()

# Apply moving average smoothing
window = 10  # number of episodes for smoothing
df_grouped["mean_reward_smooth"] = df_grouped["mean_reward"].rolling(window, min_periods=1).mean()
df_grouped["total_reward_smooth"] = df_grouped["total_reward"].rolling(window, min_periods=1).mean()
df_grouped["fairness_smooth"] = df_grouped["fairness"].rolling(window, min_periods=1).mean()

# ---------------- PLOTS ---------------- #

# 1. Mean reward per episode
plt.figure(figsize=(10, 6))
plt.plot(df_grouped["episode"], df_grouped["mean_reward_smooth"], label="Mean Reward (smoothed)")
plt.xlabel("Episode")
plt.ylabel("Mean Reward")
plt.title("Mean Reward per Episode")
plt.legend()
plt.grid(True)
plt.show()

# 2. Total reward per episode
plt.figure(figsize=(10, 6))
plt.plot(df_grouped["episode"], df_grouped["total_reward_smooth"], label="Total Reward (smoothed)", color="orange")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.legend()
plt.grid(True)
plt.show()

# 3. Fairness (Gini coefficient) per episode
plt.figure(figsize=(10, 6))
plt.plot(df_grouped["episode"], df_grouped["fairness_smooth"], label="Fairness (smoothed)", color="green")
plt.xlabel("Episode")
plt.ylabel("Fairness (Gini coefficient)")
plt.title("Fairness across Agents per Episode")
plt.legend()
plt.grid(True)
plt.show()
