import subprocess
import sys
import datetime
import os

# Auto-install required packages
required_packages = [
    "yfinance", "pandas", "numpy", "gym[classic_control]",
    "ta", "stable-baselines3", "matplotlib"
]

for package in required_packages:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except Exception as e:
        print(f"Failed to install {package}: {e}")

# Imports
import yfinance as yf
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import matplotlib.pyplot as plt

# Download AAPL 15-minute data (last 30 days)
def get_intraday_data(ticker="AAPL", interval="15m"):
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=30)
    df = yf.download(ticker, start=start.strftime('%Y-%m-%d'),
                     end=end.strftime('%Y-%m-%d'), interval=interval)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df

# Add technical indicators
def add_indicators(df):
    close = df['Close']

    rsi = RSIIndicator(close=close).rsi().squeeze()
    macd = MACD(close=close)
    macd_line = macd.macd().squeeze()
    macd_signal = macd.macd_signal().squeeze()
    bb = BollingerBands(close=close)
    bb_upper = bb.bollinger_hband().squeeze()
    bb_lower = bb.bollinger_lband().squeeze()

    df['rsi'] = pd.Series(rsi, index=df.index)
    df['macd'] = pd.Series(macd_line, index=df.index)
    df['macd_signal'] = pd.Series(macd_signal, index=df.index)
    df['bb_upper'] = pd.Series(bb_upper, index=df.index)
    df['bb_lower'] = pd.Series(bb_lower, index=df.index)

    df.dropna(inplace=True)
    return df

# Custom Gym environment
class AAPLTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(AAPLTradingEnv, self).__init__()
        self.df = df.reset_index()
        self.initial_balance = initial_balance
        self.current_step = 0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.current_step = 0
        self.trades = []
        self.net_worths = []
        return self._next_observation()

    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        obs = np.array([
            row['Open'], row['High'], row['Low'], row['Close'], row['Volume'],
            row['rsi'], row['macd'], row['macd_signal'], row['bb_upper'], row['bb_lower'],
            self.shares_held
        ])
        return obs.astype(np.float32)

    def step(self, action):
        row = self.df.iloc[self.current_step]
        price = row['Close']
        prev_net_worth = self.net_worth

        if action == 1 and self.balance >= price:
            self.shares_held += 1
            self.balance -= price
            self.trades.append(('Buy', self.current_step, price))
        elif action == 2 and self.shares_held > 0:
            self.shares_held -= 1
            self.balance += price
            self.trades.append(('Sell', self.current_step, price))

        self.current_step += 1
        self.net_worth = self.balance + self.shares_held * price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        self.net_worths.append(self.net_worth)

        reward = self.net_worth - prev_net_worth
        done = self.current_step >= len(self.df) - 1
        return self._next_observation(), reward, done, {}

    def get_performance(self):
        returns = [self.trades[i+1][2] - self.trades[i][2] for i in range(len(self.trades)-1)]
        returns = np.array(returns)
        if len(returns) == 0:
            return 0, 0, 0
        sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        total_return = self.net_worth / self.initial_balance - 1
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        return sharpe, total_return, drawdown

# Main training function
def main():
    df = get_intraday_data()
    df = add_indicators(df)
    env = AAPLTradingEnv(df)
    check_env(env)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("ppo_aapl_intraday")

    obs = env.reset()
    for _ in range(200):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break

    sharpe, total_return, drawdown = env.get_performance()

    # Output performance metrics to terminal
    print("\nPerformance Metrics:")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Total Return: {total_return*100:.2f}%")
    print(f"Max Drawdown: {drawdown*100:.2f}%")

    # Save trades to CSV
    trades_df = pd.DataFrame(env.trades, columns=["Action", "Step", "Price"])
    trades_df.to_csv("aapl_trades.csv", index=False)

    # Save net worth over time
    pd.Series(env.net_worths).to_csv("net_worth.csv", index=False)

    # Plot
    plt.plot(env.net_worths)
    plt.title("Agent Net Worth Over Time")
    plt.xlabel("Step")
    plt.ylabel("Net Worth ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("net_worth_plot.png")  # Save chart
    plt.show()

if __name__ == "__main__":
    main()
