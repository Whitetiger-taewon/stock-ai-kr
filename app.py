import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pykrx import stock
from top50_list import TOP_STOCKS
from datetime import datetime

st.set_page_config(page_title="한국 주식 AI 추천 시스템", layout="wide")

model = joblib.load("stock_model.pkl")

st.title("한국 주식 AI 추천 시스템")


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


def show_rsi_status(rsi_value):
    if rsi_value >= 70:
        st.warning(f"RSI 상태: {rsi_value:.2f} → 과열 가능성이 있습니다.")
    elif rsi_value <= 30:
        st.info(f"RSI 상태: {rsi_value:.2f} → 과매도 가능성이 있습니다.")
    else:
        st.success(f"RSI 상태: {rsi_value:.2f} → 중립 구간입니다.")


def show_macd_status(macd_value, signal_value, hist_value):
    if macd_value > signal_value:
        st.success("MACD 상태: 시그널선 위에 있어 강세 우위입니다.")
    elif macd_value < signal_value:
        st.error("MACD 상태: 시그널선 아래에 있어 약세 우위입니다.")
    else:
        st.info("MACD 상태: 시그널선과 거의 같은 수준입니다.")

    if hist_value > 0:
        st.success("MACD 히스토그램: 상승 모멘텀이 우세합니다.")
    elif hist_value < 0:
        st.error("MACD 히스토그램: 하락 모멘텀이 우세합니다.")
    else:
        st.info("MACD 히스토그램: 방향성이 강하지 않습니다.")


def get_stock_features(ticker):
    today = datetime.today().strftime("%Y%m%d")

    df = stock.get_market_ohlcv_by_date(
        "20240101",
        today,
        ticker
    )

    if df.empty:
        return None, None

    df["return"] = df["종가"].pct_change()
    df["ma5"] = df["종가"].rolling(5).mean()
    df["ma20"] = df["종가"].rolling(20).mean()

    df = add_fibonacci_features(df, window=60)
    df = add_rsi(df, period=14)
    df = add_macd(df)

    df = df.dropna()

    if df.empty:
        return None, None

    last = df.iloc[-1]

    X = pd.DataFrame([{
        "return": last["return"],
        "ma5": last["ma5"],
        "ma20": last["ma20"],
        "fib_236_diff": last["fib_236_diff"],
        "fib_382_diff": last["fib_382_diff"],
        "fib_500_diff": last["fib_500_diff"],
        "fib_618_diff": last["fib_618_diff"],
        "rsi": last["rsi"],
        "macd": last["macd"],
        "macd_signal": last["macd_signal"],
        "macd_hist": last["macd_hist"]
    }])

    return df, X


def predict_stock(ticker):
    df, X = get_stock_features(ticker)

    if df is None or X is None:
        return None

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]

    return {
        "df": df,
        "pred": pred,
        "proba": proba
    }


def nearest_fibonacci_label(last_row):
    levels = {
        "23.6%": abs(last_row["종가"] - last_row["fib_236"]),
        "38.2%": abs(last_row["종가"] - last_row["fib_382"]),
        "50.0%": abs(last_row["종가"] - last_row["fib_500"]),
        "61.8%": abs(last_row["종가"] - last_row["fib_618"]),
    }
    return min(levels, key=levels.get)


def calculate_trade_plan(last_row, proba):
    current_price = float(last_row["종가"])
    fib_236 = float(last_row["fib_236"])
    fib_382 = float(last_row["fib_382"])
    fib_500 = float(last_row["fib_500"])
    fib_618 = float(last_row["fib_618"])
    fib_high = float(last_row["fib_high"])
    fib_low = float(last_row["fib_low"])
    rsi = float(last_row["rsi"])
    macd = float(last_row["macd"])
    macd_signal = float(last_row["macd_signal"])

    # 1) 기본 매수가: 현재가와 주요 되돌림 구간 중 보수적 접근
    expected_buy = min(current_price, fib_382, fib_500)

    # 2) RSI 과열이면 매수가를 더 낮춰서 보수적 진입
    if rsi >= 70:
        expected_buy *= 0.985
    elif rsi <= 30:
        expected_buy *= 1.005

    # 3) AI 확률 기반 익절폭 설정
    if proba >= 0.70:
        take_profit_rate = 0.06
        stop_loss_rate = 0.03
    elif proba >= 0.60:
        take_profit_rate = 0.05
        stop_loss_rate = 0.025
    elif proba >= 0.50:
        take_profit_rate = 0.04
        stop_loss_rate = 0.02
    else:
        take_profit_rate = 0.03
        stop_loss_rate = 0.015

    # 4) MACD 강세면 목표가를 조금 더 높게
    macd_bonus = 1.0
    if macd > macd_signal:
        macd_bonus = 1.01
    elif macd < macd_signal:
        macd_bonus = 0.995

    # 5) 손절가 / 익절가
    stop_loss = expected_buy * (1 - stop_loss_rate)
    take_profit = expected_buy * (1 + take_profit_rate) * macd_bonus

    # 6) 예상 매도가: 최근 고점/피보나치 저항/익절가를 함께 고려
    candidate_sell_prices = [
        take_profit,
        fib_236,
        fib_high
    ]
    expected_sell = max(candidate_sell_prices)

    # 단, 지나치게 먼 목표는 제한
    upper_limit = current_price * 1.12
    expected_sell = min(expected_sell, upper_limit)

    # 7) 손익비 계산
    risk = expected_buy - stop_loss
    reward = expected_sell - expected_buy
    rr_ratio = reward / risk if risk > 0 else 0

    # 8) 손익비가 1:2보다 낮으면 목표가 보정
    min_target_sell = expected_buy + (risk * 2)
    if expected_sell < min_target_sell:
        expected_sell = min(min_target_sell, upper_limit)
        reward = expected_sell - expected_buy
        rr_ratio = reward / risk if risk > 0 else 0

    # 9) 비정상값 방지
    expected_buy = max(expected_buy, fib_low)
    stop_loss = min(stop_loss, expected_buy * 0.99)
    take_profit = max(take_profit, expected_sell)

    return {
        "current_price": round(current_price),
        "expected_buy": round(expected_buy),
        "expected_sell": round(expected_sell),
        "take_profit": round(take_profit),
        "stop_loss": round(stop_loss),
        "rr_ratio": round(rr_ratio, 2),
        "fib_236": round(fib_236),
        "fib_382": round(fib_382),
        "fib_500": round(fib_500),
        "fib_618": round(fib_618),
        "fib_high": round(fib_high),
        "fib_low": round(fib_low),
        "take_profit_rate": round(take_profit_rate * 100, 1),
        "stop_loss_rate": round(stop_loss_rate * 100, 1)
    }


st.subheader("개별 종목 AI 분석")

ticker = st.text_input("종목코드 입력", "005930")

if st.button("AI 분석 실행"):
    result = predict_stock(ticker)

    if result is None:
        st.error("종목 데이터를 가져올 수 없습니다.")
    else:
        st.write(f"상승 확률: {result['proba']:.2%}")

        if result["pred"] == 1:
            st.success("AI 판단: 상승 가능성 높음 (매수 후보)")
        else:
            st.error("AI 판단: 하락 가능성 (관망)")

        last = result["df"].iloc[-1]
        nearest_level = nearest_fibonacci_label(last)
        trade_plan = calculate_trade_plan(last, result["proba"])

        st.subheader("예상 매매 가격")
        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            st.metric("예상 매수가", f"{trade_plan['expected_buy']:,}원")
        with col_b:
            st.metric("예상 매도가", f"{trade_plan['expected_sell']:,}원")
        with col_c:
            st.metric("예상 익절가", f"{trade_plan['take_profit']:,}원")
        with col_d:
            st.metric("예상 손절가", f"{trade_plan['stop_loss']:,}원")

        st.subheader("매매 전략 요약")
        col_e, col_f, col_g = st.columns(3)

        with col_e:
            st.metric("손익비", f"1 : {trade_plan['rr_ratio']}")
        with col_f:
            st.metric("익절 기준", f"+{trade_plan['take_profit_rate']}%")
        with col_g:
            st.metric("손절 기준", f"-{trade_plan['stop_loss_rate']}%")

        st.caption(
            "매수가/매도가/익절가/손절가는 AI 직접 예측값이 아니라 "
            "AI 상승확률, RSI, MACD, 최근 고점/저점, 피보나치 구간을 반영한 규칙 기반 보조값입니다."
        )

        st.subheader("기술 분석 요약")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("현재가", f"{int(last['종가']):,}원")
            st.metric("RSI", f"{last['rsi']:.2f}")

        with col2:
            st.metric("MACD", f"{last['macd']:.2f}")
            st.metric("시그널", f"{last['macd_signal']:.2f}")

        with col3:
            st.metric("최근 60일 고점", f"{int(last['fib_high']):,}원")
            st.metric("최근 60일 저점", f"{int(last['fib_low']):,}원")

        st.subheader("RSI / MACD 상태")
        show_rsi_status(last["rsi"])
        show_macd_status(last["macd"], last["macd_signal"], last["macd_hist"])

        st.subheader("피보나치 분석")
        st.write(f"가장 가까운 피보나치 되돌림 구간: {nearest_level}")

        fib_table = pd.DataFrame({
            "구간": ["23.6%", "38.2%", "50.0%", "61.8%"],
            "가격": [
                int(last["fib_236"]),
                int(last["fib_382"]),
                int(last["fib_500"]),
                int(last["fib_618"])
            ]
        })
        st.dataframe(fib_table, use_container_width=True)

        st.subheader("최근 주가 차트")
        chart_df = result["df"][["종가", "fib_236", "fib_382", "fib_500", "fib_618"]]
        st.line_chart(chart_df)

        st.subheader("MACD 차트 데이터")
        macd_df = result["df"][["macd", "macd_signal", "macd_hist"]].tail(30)
        st.line_chart(macd_df)

        st.subheader("최근 주가 데이터")
        st.dataframe(result["df"].tail(10), use_container_width=True)


st.subheader("오늘의 AI 추천 종목 TOP5")

if st.button("추천 종목 보기"):
    results = []

    for code, name in TOP_STOCKS.items():
        result = predict_stock(code)

        if result is not None:
            last = result["df"].iloc[-1]
            trade_plan = calculate_trade_plan(last, result["proba"])

            results.append({
                "종목명": name,
                "종목코드": code,
                "상승확률": result["proba"],
                "예상매수가": trade_plan["expected_buy"],
                "예상매도가": trade_plan["expected_sell"]
            })

    if len(results) == 0:
        st.error("추천 종목을 계산할 수 없습니다.")
    else:
        result_df = pd.DataFrame(results)
        top5_df = result_df.sort_values(by="상승확률", ascending=False).head(5).copy()

        display_df = top5_df.copy()
        display_df["상승확률"] = display_df["상승확률"].apply(lambda x: f"{x:.2%}")
        display_df["예상매수가"] = display_df["예상매수가"].apply(lambda x: f"{x:,}원")
        display_df["예상매도가"] = display_df["예상매도가"].apply(lambda x: f"{x:,}원")

        st.dataframe(display_df, use_container_width=True)

        st.subheader("추천 종목 상승확률 그래프")
        chart_data = top5_df.set_index("종목명")["상승확률"]
        st.bar_chart(chart_data)