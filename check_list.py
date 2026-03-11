from top50_list import TOP_STOCKS

print("분석 대상 종목 수:", len(TOP_STOCKS))
print()

for code, name in TOP_STOCKS.items():
    print(code, name)