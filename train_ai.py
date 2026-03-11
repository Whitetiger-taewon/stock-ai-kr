import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 데이터 읽기
df = pd.read_csv("samsung_stock.csv")

# 목표값 생성 (다음날 상승 여부)
df["target"] = (df["종가"].shift(-1) > df["종가"]).astype(int)

df = df.dropna()

# 사용할 특징
features = [
    "return",
    "ma5",
    "ma20",
    "fib_236_diff",
    "fib_382_diff",
    "fib_500_diff",
    "fib_618_diff",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist"
]

X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("AI 모델 정확도:", accuracy)

joblib.dump(model, "stock_model.pkl")
print("AI 모델 저장 완료")