from pykrx import stock

# 삼성전자 종목코드
ticker = "005930"

# 주가 데이터 가져오기
df = stock.get_market_ohlcv_by_date(
    "20240101",
    "20240601",
    ticker
)

print(df.tail())