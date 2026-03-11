import pandas as pd
import joblib
from pykrx import stock
from top50_list import TOP_STOCKS

model = joblib.load("stock_model.pkl")

results = []

for code, name in TOP_STOCKS.items():

    df = stock.get_market_ohlcv_by_date(
        "20230101",
        "20240101",
        code
    )

    df["return"] = df["종가"].pct_change()
    df["ma5"] = df["종가"].rolling(5).mean()
    df["ma20"] = df["종가"].rolling(20).mean()

    df = df.dropna()

    last = df.iloc[-1]

    X = pd.DataFrame([{
        "return": last["return"],
        "ma5": last["ma5"],
        "ma20": last["ma20"]
    }])

    prob = model.predict_proba(X)[0][1]

    results.append((name, prob))

results.sort(key=lambda x: x[1], reverse=True)

print("\nAI 추천 종목 TOP5\n")

for i in range(5):
    name, prob = results[i]
    print(f"{i+1}. {name}  상승확률: {prob:.2%}")