import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
from stable_baselines3.common.callbacks import BaseCallback
warnings.filterwarnings("ignore")

from data.alpaca_downloader import fetch_data_alpaca
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from env.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "V", "WMT", "PG", "JNJ", "HD", "BAC", "MA", "UNH"]
data_list = []

print("📥 Fetching data for:", tickers)
for tic in tickers:
    try:
        df = fetch_data_alpaca(tic, "2018-01-01", "2023-12-31")
        
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['rsi_14'] = df['close'].diff().rolling(window=14).apply(lambda x: 100 - (100 / (1 + (x[x > 0].sum() / -x[x < 0].sum()))))
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['bb_upper'] = df['close'].rolling(window=20).mean() + 2 * df['close'].rolling(window=20).std()
        df['bb_lower'] = df['close'].rolling(window=20).mean() - 2 * df['close'].rolling(window=20).std()
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        

        df['turbulence'] = df['close'].pct_change().rolling(window=20).std()
        

        df['sma_20'].fillna(method='bfill', inplace=True)
        df['sma_50'].fillna(method='bfill', inplace=True)
        df['rsi_14'].fillna(50, inplace=True)  
        df['macd'].fillna(0, inplace=True)
        df['macd_signal'].fillna(0, inplace=True)
        df['bb_upper'].fillna(method='bfill', inplace=True)
        df['bb_lower'].fillna(method='bfill', inplace=True)
        df['volume_sma'].fillna(method='bfill', inplace=True)
        df['turbulence'].fillna(0, inplace=True)
        

        required_indicators = ['macd', 'rsi_14', 'bb_upper', 'bb_lower', 'volume_sma', 'turbulence']
        for indicator in required_indicators:
            if indicator not in df.columns:
                raise ValueError(f"Missing required indicator: {indicator}")
        

        tech_indicators = df[required_indicators].copy()
        
        fe = FeatureEngineer(
            use_technical_indicator=False,  
            tech_indicator_list=required_indicators,
            use_vix=False,
            use_turbulence=False,  
            user_defined_feature=False
        )
        

        df = fe.preprocess_data(df)
        

        for indicator in required_indicators:
            df[indicator] = tech_indicators[indicator]
        
        data_list.append(df)
        print(".", end="", flush=True)  
        time.sleep(1)  
        
    except Exception as e:
        print(f"\n❌ Error processing {tic}: {str(e)}")
        continue

print("\n")  

if not data_list:
    raise ValueError("No data was successfully processed. Please check the errors above.")

df_all = pd.concat(data_list)
df_all.sort_values(by=["date", "tic"], inplace=True)
df_all["date"] = df_all["date"].astype(str)

valid_dates = df_all.groupby("date").count()["tic"]
valid_dates = valid_dates[valid_dates == len(tickers)].index
df_all = df_all[df_all["date"].isin(valid_dates)]


if df_all.isnull().any().any():
    print("Warning: NaN values found in dataset. Filling with appropriate values...")
    df_all.fillna(method='ffill', inplace=True)
    df_all.fillna(method='bfill', inplace=True)  

train = data_split(df_all, start="2018-01-01", end="2022-12-31")
test = data_split(df_all, start="2023-01-01", end="2023-12-31")

tech_indicators = ['macd', 'rsi_14', 'bb_upper', 'bb_lower', 'volume_sma']
stock_dim = len(tickers)
state_space = 1 + 2 * stock_dim + len(tech_indicators) * stock_dim

env_kwargs = {
    "hmax": 100,
    "initial_amount": 100000,
    "buy_cost_pct": [0.001] * stock_dim,
    "sell_cost_pct": [0.001] * stock_dim,
    "state_space": state_space,
    "stock_dim": stock_dim,
    "tech_indicator_list": tech_indicators,
    "action_space": stock_dim,
    "reward_scaling": 1e-2,
    "num_stock_shares": [0] * stock_dim,
    "turbulence_threshold": 30,
    "risk_indicator_col": "turbulence"
}

env_train = StockTradingEnv(df=train, **env_kwargs)
env_test = StockTradingEnv(df=test, **env_kwargs)

agent = DRLAgent(env=env_train)
ppo_model = agent.get_model("ppo")

ppo_model.learning_rate = 3e-4
ppo_model.n_steps = 2048
ppo_model.batch_size = 64
ppo_model.n_epochs = 10
ppo_model.gamma = 0.99
ppo_model.gae_lambda = 0.95
ppo_model.clip_range = lambda _: 0.2  
ppo_model.ent_coef = 0.01

print("Starting training...")
trained_ppo = agent.train_model(
    model=ppo_model,
    tb_log_name="multi_ticker_ppo",
    total_timesteps=500_000
)

print("Making predictions...")
df_account_value, df_actions = agent.DRL_prediction(
    model=trained_ppo,
    environment=env_test
)

df_account_value['account_value'] = df_account_value['account_value'].astype(float)
df_account_value['date'] = pd.to_datetime(df_account_value['date'])
df_account_value.set_index('date', inplace=True)


plt.figure(figsize=(12, 6))
plt.plot(df_account_value['account_value'], label="Portfolio Value")
plt.title('💼 Multi-Ticker PNL Curve (PPO)')
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("pnl_multi_ticker_ppo.png")
plt.show()

daily_returns = df_account_value['account_value'].pct_change().dropna()
sharpe_ratio = (252 ** 0.5) * daily_returns.mean() / daily_returns.std()
print(f"✅ Final Portfolio Value: ${df_account_value['account_value'][-1]:,.2f}")
print(f"📊 Sharpe Ratio: {sharpe_ratio:.3f}")

annual_return = (df_account_value['account_value'][-1] / df_account_value['account_value'][0]) ** (252/len(df_account_value)) - 1
max_drawdown = (df_account_value['account_value'] / df_account_value['account_value'].cummax() - 1).min()
print(f"📈 Annual Return: {annual_return:.2%}")
print(f"📉 Maximum Drawdown: {max_drawdown:.2%}")
