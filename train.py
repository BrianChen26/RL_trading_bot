import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from data.alpaca_downloader import fetch_data_alpaca
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent

# ============================
# 🔧 Step 1: 数据处理
# ============================
print("📥 Fetching data...")
df = fetch_data_alpaca("AMZN", "2019-01-01", "2023-01-01")

fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=['macd', 'rsi_30', 'cci_30', 'dx_30'],
    use_vix=False,
    use_turbulence=False,
    user_defined_feature=False
)
df = fe.preprocess_data(df)
df["date"] = df["date"].astype(str)

train = data_split(df, start='2019-01-01', end='2022-01-01')
test = data_split(df, start='2022-01-01', end='2023-01-01')

# ============================
# 📊 Step 2: 环境参数设置
# ============================
tech_indicators = ['macd', 'rsi_30', 'cci_30', 'dx_30']
stock_dimension = len(train.tic.unique())
state_space = 1 + 2 * stock_dimension + len(tech_indicators) * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 100000,
    "buy_cost_pct": [0.001] * stock_dimension,
    "sell_cost_pct": [0.001] * stock_dimension,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": tech_indicators,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
    "num_stock_shares": [0] * stock_dimension
}

env_train = StockTradingEnv(df=train, **env_kwargs)
env_test = StockTradingEnv(df=test, **env_kwargs)

# ============================
# 🤖 Step 3: PPO 训练
# ============================
print("🚀 Training PPO agent...")
agent = DRLAgent(env=env_train)
ppo_model = agent.get_model("ppo")

trained_ppo = agent.train_model(
    model=ppo_model, 
    tb_log_name='ppo', 
    total_timesteps=100000
)

# ============================
# 📈 Step 4: 回测与画图
# ============================
print("🔍 Evaluating on test set...")
df_account_value, df_actions = agent.DRL_prediction(
    model=trained_ppo, 
    environment=env_test
)

df_account_value['account_value'] = df_account_value['account_value'].astype(float)
df_account_value['date'] = pd.to_datetime(df_account_value['date'])
df_account_value.set_index('date', inplace=True)

# Plot Portfolio Curve
plt.figure(figsize=(10, 5))
plt.plot(df_account_value['account_value'], label="Portfolio Value")
plt.title('📈 PNL Curve (PPO on AAPL)')
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("pnl_ppo.png")
plt.show()

# Calculate Sharpe Ratio
daily_returns = df_account_value['account_value'].pct_change().dropna()
sharpe_ratio = (252 ** 0.5) * daily_returns.mean() / daily_returns.std()
print(f"✅ Final Portfolio Value: ${df_account_value['account_value'][-1]:,.2f}")
print(f"📊 Sharpe Ratio: {sharpe_ratio:.3f}")
