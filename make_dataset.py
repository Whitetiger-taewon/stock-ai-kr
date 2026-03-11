from pykrx import stock
import pandas as pd
import numpy as np


def add_fibonacci_features(df, window=60):
    df = df.copy()

    rolling_high = df["고가"].rolling(window).max()
    rolling_low = df["저가"].rolling(window).min()
    price_range = rolling_high - rolling_low
    price_range = price_range.replace(0, np.nan)

    df["fib_high"] = rolling_high
    df["fib_low"] = rolling_low

    df["fib_236"] = rolling_high - price_range * 0.236
    df["fib_382"] = rolling_high - price_range * 0.382
    df["fib_500"] = rolling_high - price_range * 0.500
    df["fib_618"] = rolling_high - price_range * 0.618

    df["fib_236_diff"] = (df["종가"] - df["fib_236"]) / df["종가"]
    df["fib_382_diff"] = (df["종가"] - df["fib_382"]) / df["종가"]
    df["fib_500_diff"] = (df["종가"] - df["fib_500"]) / df["종가"]
    df["fib_618_diff"] = (df["종가"] - df["fib_618"]) / df["종가"]

    return df


def add_rsi(df, period=14):
    df = df.copy()

    delta = df["종가"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    return df


def add_macd(df, short=12, long=26, signal=9):
    df = df.copy()

    ema_short = df["종가"].ewm(span=short, adjust=False).mean()
    ema_long = df["종가"].ewm(span=long, adjust=False).mean()

    df["macd"] = ema_short - ema_long
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    return df


ticker = "005930"

df = stock.get_market_ohlcv_by_date(
    "20150101",
    "20240101",
    ticker
)

df["return"] = df["종가"].pct_change()
df["ma5"] = df["종가"].rolling(5).mean()
df["ma20"] = df["종가"].rolling(20).mean()

df = add_fibonacci_features(df, window=60)
df = add_rsi(df, period=14)
df = add_macd(df)

df = df.dropna()

print(df.tail())

df.to_csv("samsung_stock.csv", index=False, encoding="utf-8-sig")

print("데이터 저장 완료")