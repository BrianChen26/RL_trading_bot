# RL Trading Bot

A Reinforcement Learning (RL) based algorithmic trading system that uses Proximal Policy Optimization (PPO) to trade multiple stocks. The system leverages FinRL framework and Alpaca API for data and trading execution.

## Project Overview

This trading bot implements a multi-ticker strategy using Deep Reinforcement Learning. The bot is trained on historical stock data and can make trading decisions for a portfolio of stocks based on technical indicators and market conditions.

### Key Features

- **Multi-Ticker Trading**: Supports trading across multiple stocks simultaneously
- **PPO Algorithm**: Uses Proximal Policy Optimization for stable training
- **Technical Indicators**: Incorporates MACD, RSI, Bollinger Bands, and more
- **Risk Management**: Implements turbulence-based risk controls
- **Performance Metrics**: Tracks Sharpe ratio, returns, and drawdowns
- **Real-time Data**: Uses Alpaca API for market data

## Project Structure

```
RL_trading_bot/
├── data/
│   └── alpaca_downloader.py      # Data fetching from Alpaca API
├── env/
│   └── env_stocktrading.py       # Custom trading environment
├── utils/
│   └── helpers.py                # Utility functions
├── trained_models/
│   └── ppo_model.zip             # Pre-trained PPO model
├── multi.py                      # Multi-ticker training and evaluation
├── train.py                      # Single-ticker training script
├── backtest.py                   # Backtesting functionality
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── pnl_ppo.png                   # Single-ticker performance plot
└── pnl_multi_ticker_ppo.png      # Multi-ticker performance plot
```

## Guideline 

### Prerequisites

- Python 3.8+
- Alpaca API credentials
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/BrianChen26/RL_trading_bot.git
   cd RL_trading_bot
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   ALPACA_API_KEY= your_api_key
   ALPACA_SECRET_KEY= your_secret_key
   ```

## Usage

### Multi-Ticker Trading (Recommended)

The main script for multi-ticker trading:

```bash
python multi.py
```

This script:
- Fetches data for 15 major stocks (AAPL, MSFT, GOOGL, etc.)
- Trains a PPO model on 2018-2022 data
- Evaluates performance on 2023 data
- Generates performance plots and metrics

### Single-Ticker Training

For training on a single stock:

```bash
python train.py
```

### Backtesting

To run backtests on trained models:

```bash
python backtest.py
```

## Trading Strategy

### Technical Indicators Used

- **MACD**: Moving Average Convergence Divergence
- **RSI**: Relative Strength Index (14-period)
- **Bollinger Bands**: Upper and lower bands (20-period)
- **SMA**: Simple Moving Averages (20 and 50-period)
- **Volume SMA**: Volume moving average
- **Turbulence**: Market volatility indicator

### Risk Management

- **Turbulence Threshold**: Automatic position liquidation during high volatility
- **Transaction Costs**: Realistic buy/sell costs (0.1%)
- **Position Limits**: Maximum position size controls
- **Reward Shaping**: Multi-component reward function including:
  - Raw returns
  - Risk penalties
  - Trading costs
  - Turnover penalties
  - Concentration penalties

## Performance Results

Based on the multi-ticker backtest (2023):

- **Portfolio Growth**: ~60% increase from $100k to $160k+
- **Sharpe Ratio**: Optimized for risk-adjusted returns
- **Drawdown Management**: Controlled volatility through turbulence monitoring
- **Multi-Asset Diversification**: Reduced concentration risk



## Important Notes

1. **Paper Trading**: This system is designed for educational and research purposes. Use paper trading accounts for testing.

2. **API Limits**: Be aware of Alpaca API rate limits when fetching data.


## 📄 License

This project is for educational purposes. Please ensure compliance with your local financial regulations before using for actual trading.

## Dependencies

- **FinRL**: Financial Reinforcement Learning framework
- **Stable-Baselines3**: RL algorithms implementation
- **Alpaca-py**: Alpaca API client
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib**: Plotting and visualization
- **Gymnasium**: RL environment interface


**Disclaimer**: This software is for educational purposes only. Trading involves substantial risk and is not suitable for all investors. Past performance does not guarantee future results.
