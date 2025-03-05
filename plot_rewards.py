import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.linear_model import TheilSenRegressor


# Load the rewards from the .npy file
# rewards = np.load('ppo_flappy_rewards.npy')
rewards = np.load('ppo_snake_rewards.npy')
# rewards = np.load('ppo_snake_losses.npy')
# rewards = np.load('ppo_snake_simple_rewards.npy')

# Compute variation metrics
std_dev = np.std(rewards)  # Standard Deviation
iqr = np.percentile(rewards, 75) - np.percentile(rewards, 25)  # Interquartile Range
cv = std_dev / np.mean(rewards) if np.mean(rewards) != 0 else float('inf')  # Coefficient of Variation
autocorr = np.corrcoef(rewards[:-1], rewards[1:])[0, 1] if len(rewards) > 1 else None  # Autocorrelation (Lag-1)
reward_range = np.max(rewards) - np.min(rewards)  # Range

# Print results
print(f"Standard Deviation: {std_dev}")
print(f"Interquartile Range (IQR): {iqr}")
print(f"Coefficient of Variation (CV): {cv}")
print(f"Autocorrelation (Lag-1): {autocorr}")
print(f"Range: {reward_range}")

# Define rolling window size
rolling_window = max(len(rewards) // 100, 10)  # Adjust for better peak sensitivity

# Compute rolling max to emphasize peaks
rolling_max = pd.Series(rewards).rolling(window=rolling_window, min_periods=1).max()

# Apply an exponentially weighted moving average (EWMA) for smoothing
smoothed_rewards = rolling_max.ewm(span=rolling_window, adjust=False).mean()

filter_size = len(rewards) // 10
if filter_size % 2 == 0:
    filter_size += 1
smoothed_savgol = savgol_filter(rewards, filter_size, 3)

# Calculate the trend line (linear fit)
x = np.arange(len(smoothed_savgol)).reshape(-1, 1)
regressor = TheilSenRegressor()
regressor.fit(x, smoothed_savgol)
trend = regressor.predict(x)
is_the_model_learning = regressor.coef_[0] > 0
print(f"Is the model learning? {is_the_model_learning}")
print(f"Model learning rate: {regressor.coef_[0]}")

# Calulate the trend line of the last 20% episodes
x_last = x[-int(len(x) * 0.2):]
rewards_last = smoothed_savgol[-int(len(rewards) * 0.2):]
regressor_last = TheilSenRegressor()
regressor_last.fit(x_last, rewards_last)
trend_last = regressor_last.predict(x_last)
is_the_model_learning_last = regressor_last.coef_[0] > 0
print(f"Is the model learning in the last 20%? {is_the_model_learning_last}")
print(f"Model learning rate in the last 20%: {regressor_last.coef_[0]}")

# Plot the original and smoothed rewards
# plt.plot(loss, alpha=0.5, label="Loss")
plt.plot(rewards, alpha=0.5, label="Raw Rewards")
plt.plot(smoothed_rewards, 'r', label="Optimistic Smoothed Rewards")
plt.plot(smoothed_savgol, 'g', label="Smoothed Rewards")
plt.plot(trend, 'b', label="Trend Line")
plt.plot(x_last, trend_last, 'm', label="Trend Line (Last 20%)")
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Snake PPO Rewards')
plt.legend()

# plt.savefig('ppo_snake_rewards.png')

plt.show()
