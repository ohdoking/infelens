# train.py
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1) 데이터 로딩 & 병합
data_dir = "path/to/csv_folder"   # <- CSV 파일들이 모여 있는 폴더 경로
all_files = glob.glob(os.path.join(data_dir, "*.csv"))
df_list = [pd.read_csv(f) for f in all_files]
df = pd.concat(df_list, ignore_index=True)
print(f"Loaded {len(df)} rows from {len(all_files)} files")

# 2) 불필요한 컬럼 제거
drop_cols = [
    # 인덱스나 메타 정보
    "Unnamed: 0", "Unnamed: 0.1", "Unnamed: 0.2", "index",
    # 타임스탬프/본문 텍스트
    "created_at", "start_time", "end_time", "prompt", "response",
    # 기타 string 특성 (필요시 남겨도 됨)
    "text_standard", "type"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# 3) 타겟 설정
TARGET = "energy_consumption_llm_gpu"
assert TARGET in df.columns, f"{TARGET} 컬럼이 없습니다."
y = df[TARGET]
X = df.drop(columns=[TARGET])

# 4) 특성 자동 분류
# 문자열 타입은 범주형, 그 외는 수치형으로 간주
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=["number"]).columns.tolist()

print("Categorical features:", cat_cols)
print("Numeric features:", len(num_cols), "columns")

# 5) 전처리 & 모델 파이프라인 구성
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols)
])

regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", regressor)
])

# 6) 학습/검증 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7) 모델 학습
print("Training model...")
pipeline.fit(X_train, y_train)

# 8) 평가
y_pred = pipeline.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R2:", r2_score(y_test, y_pred))

# 9) 모델 저장
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "rf_energy_gpu.pkl")
joblib.dump(pipeline, model_path)
print(f"Model saved to {model_path}")

