import os
import pandas as pd
from datetime import datetime
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")


client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

def fetch_data_alpaca(symbol="AAPL", start="2019-01-01", end="2023-01-01"):
    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=pd.Timestamp(start, tz='America/New_York'),
        end=pd.Timestamp(end, tz='America/New_York')
    )

    bars = client.get_stock_bars(request_params).df

    if isinstance(bars.index, pd.MultiIndex):
        bars.index = bars.index.get_level_values("timestamp")

    df = bars.copy()

    df = df.rename(columns={
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume"
    })

    df["tic"] = symbol
    df["date"] = pd.to_datetime(df.index).date
    df = df[["date", "open", "high", "low", "close", "volume", "tic"]]
    df = df.sort_values("date").reset_index(drop=True)
    return df



if __name__ == "__main__":
    df = fetch_data_alpaca()
    print(df.head())
