import os
import pandas as pd
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3 import PPO
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from finrl.meta.preprocessor.preprocessors import data_split

# Load environment variables
load_dotenv()


def load_test_data():
    API_KEY = os.getenv("ALPACA_API_KEY")
    API_SECRET = os.getenv("ALPACA_SECRET_KEY")

    client = StockHistoricalDataClient(API_KEY, API_SECRET)

    tickers = ["AAPL", "MSFT", "GOOG"]
    df_list = []

    for ticker in tickers:
        request_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start="2021-01-01",
            end="2023-01-01"
        )
        bars = client.get_stock_bars(request_params).df.reset_index()
        bars = bars.rename(columns={"symbol": "tic", "timestamp": "date"})
        bars["date"] = pd.to_datetime(bars["date"]).dt.tz_localize(None)
        bars = bars[["date", "open", "high", "low", "close", "volume", "tic"]]

        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=["macd", "rsi_30", "cci_30", "dx_30"],
            use_vix=False,
            use_turbulence=False,
            user_defined_feature=False,
        )
        processed = fe.preprocess_data(bars)
        df_list.append(processed)

    df_all = pd.concat(df_list)
    df_all = df_all.sort_values(by=["date", "tic"]).reset_index(drop=True)

    # ✅ Keep only dates that have all tickers
    valid_dates = df_all.groupby("date")["tic"].nunique()
    df_all = df_all[df_all["date"].isin(valid_dates[valid_dates == len(tickers)].index)]

    print("✅ Sample processed data:")
    print(df_all.head(10))
    return df_all


def run_backtest(model, env):
    obs, _ = env.reset()
    done = False
    trade_count = 0
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)

        trade_count += np.count_nonzero(action)

    return env.asset_memory, trade_count



if __name__ == "__main__":
    df = load_test_data()
    test_df = data_split(df, start='2022-01-01', end='2023-01-01')


    stock_dim = len(test_df["tic"].unique())
    tech_indicators = ["macd", "rsi_30", "cci_30", "dx_30"]
    state_space = 1 + 2 * stock_dim + len(tech_indicators) * stock_dim
    action_space = stock_dim
    num_stock_shares = [0] * stock_dim

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 100000,
        "buy_cost_pct": [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "tech_indicator_list": tech_indicators,
        "action_space": action_space,
        "print_verbosity": 1
    }

    print(f"✅ Env initialized with {stock_dim} stocks and {len(test_df)} data rows.")

    test_env = StockTradingEnv(
        df=test_df,
        stock_dim=stock_dim,
        num_stock_shares=num_stock_shares,
        **env_kwargs
    )

    model = PPO.load("trained_models/ppo_model")


    asset_memory, total_trades = run_backtest(model, test_env)

    print(f"\nFinal Portfolio Value: {asset_memory[-1]}")
    print(f"Total Trades: {total_trades}")

    plt.plot(asset_memory)
    plt.xlabel("Time Step")
    plt.ylabel("Total Asset Value")
    plt.title("Backtest Result")
    plt.grid()
    plt.tight_layout()
    plt.show()
