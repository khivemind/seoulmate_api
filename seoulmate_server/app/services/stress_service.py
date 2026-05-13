from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "app/models/stress_model.pkl"
HISTORY_PATH = BASE_DIR / "app/data/stress_history_standard.csv"

_model_package = None
_history = None


STRESS_GRADE_BINS = [-0.1, 20, 40, 60, 80, 100]
STRESS_GRADE_LABELS = [1, 2, 3, 4, 5]


def load_model_package():
    global _model_package

    if _model_package is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"stress 모델 파일이 없음: {MODEL_PATH}")

        obj = joblib.load(MODEL_PATH)

        if isinstance(obj, dict):
            package = obj
        else:
            package = {"model": obj}

        model = package.get("model")

        if model is None:
            raise ValueError(
                f"stress_model.pkl에 'model'이 없음. 현재 keys: {list(package.keys())}"
            )

        # =====================================================
        # feature names 통일
        # 네 모델 파일은 feature_cols를 사용함
        # =====================================================
        if "feature_names" not in package:
            if "feature_cols" in package:
                package["feature_names"] = list(package["feature_cols"])
            elif hasattr(model, "feature_name_"):
                package["feature_names"] = list(model.feature_name_)
            elif hasattr(model, "feature_names_in_"):
                package["feature_names"] = list(model.feature_names_in_)
            elif hasattr(model, "booster_"):
                package["feature_names"] = list(model.booster_.feature_name())
            else:
                raise ValueError(
                    "feature_names 또는 feature_cols를 찾을 수 없음. "
                    f"현재 package keys: {list(package.keys())}"
                )

        # =====================================================
        # categorical columns 통일
        # 네 모델 파일은 categorical_features를 사용함
        # =====================================================
        if "cat_cols" not in package:
            if "categorical_features" in package:
                package["cat_cols"] = list(package["categorical_features"])
            else:
                package["cat_cols"] = []

        print("[stress model loaded]")
        print("model type:", type(package["model"]))
        print("package keys:", list(package.keys()))
        print("feature count:", len(package["feature_names"]))
        print("feature sample:", package["feature_names"][:30])
        print("cat_cols:", package["cat_cols"])
        print("target_col:", package.get("target_col"))
        print("target_type:", package.get("target_type"))

        _model_package = package

    return _model_package


def load_history():
    global _history

    if _history is None:
        if not HISTORY_PATH.exists():
            raise FileNotFoundError(f"stress history 파일이 없음: {HISTORY_PATH}")

        df = pd.read_csv(HISTORY_PATH, dtype=str)
        df.columns = df.columns.str.strip()

        required_cols = [
            "code",
            "gu",
            "dong",
            "year",
            "month",
            "소음 평균(dB)",
            "소음 최대(dB)",
            "소음 최소(dB)",
            "진동(x) 평균(mm/s)",
            "진동(y) 평균(mm/s)",
            "진동(z) 평균(mm/s)",
        ]

        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            raise ValueError(
                f"stress_history_standard.csv 필수 컬럼 없음: {missing}. "
                f"현재 컬럼: {df.columns.tolist()}"
            )

        df["code"] = df["code"].astype(str).str.strip()
        df["gu"] = df["gu"].astype(str).str.strip()
        df["dong"] = df["dong"].astype(str).str.strip()
        df["year"] = df["year"].astype(int)
        df["month"] = df["month"].astype(int)

        numeric_cols = [
            "hour",
            "day",
            "dayofweek",
            "is_weekend",
            "소음 평균(dB)",
            "소음 최대(dB)",
            "소음 최소(dB)",
            "진동(x) 평균(mm/s)",
            "진동(y) 평균(mm/s)",
            "진동(z) 평균(mm/s)",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        _history = df

        print("[stress history loaded]")
        print("shape:", _history.shape)
        print("columns:", _history.columns.tolist())

    return _history



def add_noise_stress_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    noise_cols = [
        "소음 평균(dB)",
        "소음 최대(dB)",
        "소음 최소(dB)",
    ]

    for col in noise_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1. 소음 평균이 60dB를 넘는 정도
    df["noise_excess"] = (
        df["소음 평균(dB)"] - 60
    ).clip(lower=0)

    # 2. 소음 변동폭
    df["noise_variability"] = (
        df["소음 최대(dB)"] - df["소음 최소(dB)"]
    )

    # 3. 순간 피크와 평균의 차이
    df["noise_peak_diff"] = (
        df["소음 최대(dB)"] - df["소음 평균(dB)"]
    )

    # 4. raw 스트레스 점수
    df["_noise_stress_raw"] = (
        df["noise_excess"] +
        df["noise_variability"] * 0.3 +
        df["noise_peak_diff"] * 0.5
    )

    # 5. 백분위 기반 0~100 점수
    df["noise_stress_score"] = (
        df["_noise_stress_raw"].rank(pct=True) * 100
    )

    return df



def make_monthly_features(year: int, month: int) -> pd.DataFrame:
    history = load_history().copy()

    # =====================================================
    # 1. 타입 정리
    # =====================================================
    history["code"] = history["code"].astype(str).str.strip()
    history["gu"] = history["gu"].astype(str).str.strip()
    history["dong"] = history["dong"].astype(str).str.strip()
    history["year"] = pd.to_numeric(history["year"], errors="coerce").astype(int)
    history["month"] = pd.to_numeric(history["month"], errors="coerce").astype(int)

    numeric_cols = [
        "소음 평균(dB)",
        "소음 최대(dB)",
        "소음 최소(dB)",
        "진동(x) 평균(mm/s)",
        "진동(y) 평균(mm/s)",
        "진동(z) 평균(mm/s)",
    ]

    for col in numeric_cols:
        if col in history.columns:
            history[col] = pd.to_numeric(history[col], errors="coerce")

    # =====================================================
    # 2. 노트북과 같은 noise_stress_score 생성
    # =====================================================
    history = add_noise_stress_score(history)

    # =====================================================
    # 3. 월별 집계
    # 모델 피처에 필요한 값:
    # stress_lag_1
    # stress_lag_2
    # noise_avg_lag_1
    # stress_roll_std_3
    # =====================================================
    monthly = (
        history
        .groupby(["code", "gu", "dong", "year", "month"], as_index=False)
        .agg(
            noise_stress_score=("noise_stress_score", "mean"),
            noise_avg=("소음 평균(dB)", "mean"),
        )
    )

    monthly["date_key"] = pd.to_datetime(
        monthly["year"].astype(str) + "-"
        + monthly["month"].astype(str).str.zfill(2) + "-01"
    )

    target_date = pd.Timestamp(year=int(year), month=int(month), day=1)

    # =====================================================
    # 4. target month row가 없을 때도 예측 가능하게 target row 생성
    # 예: 2026-05 데이터가 없어도 2026-04까지로 5월 예측 가능
    # =====================================================
    dong_master = (
        monthly[["code", "gu", "dong"]]
        .drop_duplicates()
        .copy()
    )

    target_rows = dong_master.copy()
    target_rows["year"] = int(year)
    target_rows["month"] = int(month)
    target_rows["date_key"] = target_date
    target_rows["noise_stress_score"] = np.nan
    target_rows["noise_avg"] = np.nan

    monthly_without_target = monthly[
        monthly["date_key"] != target_date
    ].copy()

    monthly_all = pd.concat(
        [monthly_without_target, target_rows],
        ignore_index=True
    )

    monthly_all = (
        monthly_all
        .drop_duplicates(subset=["code", "year", "month"], keep="last")
        .sort_values(["code", "date_key"])
        .reset_index(drop=True)
    )

    # =====================================================
    # 5. lag / rolling 생성
    # =====================================================
    monthly_all["stress_lag_1"] = (
        monthly_all
        .groupby("code")["noise_stress_score"]
        .shift(1)
    )

    monthly_all["stress_lag_2"] = (
        monthly_all
        .groupby("code")["noise_stress_score"]
        .shift(2)
    )

    monthly_all["noise_avg_lag_1"] = (
        monthly_all
        .groupby("code")["noise_avg"]
        .shift(1)
    )

    monthly_all["stress_roll_std_3"] = (
        monthly_all
        .groupby("code")["noise_stress_score"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=2).std())
    )

    monthly_all["stress_roll_std_3"] = (
        monthly_all["stress_roll_std_3"].fillna(0)
    )

    # =====================================================
    # 6. 계절성 피처
    # =====================================================
    monthly_all["month_num"] = monthly_all["month"].astype(int)

    monthly_all["month_sin"] = np.sin(
        2 * np.pi * monthly_all["month_num"] / 12
    )

    monthly_all["month_cos"] = np.cos(
        2 * np.pi * monthly_all["month_num"] / 12
    )

    # =====================================================
    # 7. 모델 학습 당시 컬럼명으로 맞추기
    # categorical_features: ['자치구', '행정동']
    # =====================================================
    monthly_all["자치구"] = monthly_all["gu"]
    monthly_all["행정동"] = monthly_all["dong"]

    # =====================================================
    # 8. 예측 대상 월만 추출
    # =====================================================
    target = monthly_all[
        (monthly_all["year"] == int(year)) &
        (monthly_all["month"] == int(month))
    ].copy()

    # lag 없으면 예측 불가라 제거
    target = target.dropna(
        subset=[
            "stress_lag_1",
            "stress_lag_2",
            "noise_avg_lag_1",
        ]
    ).copy()

    return target


def build_X(feature_df: pd.DataFrame, feature_names: list[str], cat_cols: list[str]) -> pd.DataFrame:
    X = feature_df.copy()

    # 모델 입력에서 식별자는 보통 제외되지만,
    # feature_names에 있으면 그대로 들어가게 둔다.
    missing_features = [c for c in feature_names if c not in X.columns]

    if missing_features:
        warnings.warn(
            f"모델 feature_names 중 현재 feature_df에 없는 컬럼이 있음. "
            f"0으로 채움: {missing_features[:30]}"
        )

        for col in missing_features:
            X[col] = 0

    X = X.reindex(columns=feature_names)

    for col in X.columns:
        if col in cat_cols:
            X[col] = X[col].astype("category")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    return X


def predict_score(feature_df: pd.DataFrame) -> pd.DataFrame:
    package = load_model_package()

    model = package["model"]

    # 네 모델 패키지 기준
    feature_names = list(package.get("feature_names", package.get("feature_cols")))
    cat_cols = list(package.get("cat_cols", package.get("categorical_features", [])))

    if feature_df.empty:
        feature_df["score"] = []
        return feature_df

    X = feature_df.reindex(columns=feature_names).copy()

    for col in X.columns:
        if col in cat_cols:
            X[col] = X[col].astype("category")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    predicted_change = model.predict(X)

    result = feature_df.copy()
    result["predicted_change"] = predicted_change

    # 핵심:
    # 최종 소음 스트레스 점수 = 직전 달 점수 + 예측 변화량
    result["score"] = (
        result["stress_lag_1"] + result["predicted_change"]
    ).clip(0, 100)

    return result


def add_grade(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["grade"] = []
        return df

    df = df.copy()

    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).clip(0, 100)

    df["grade"] = pd.cut(
        df["score"],
        bins=STRESS_GRADE_BINS,
        labels=STRESS_GRADE_LABELS,
        include_lowest=True,
    ).astype(int)

    return df


def get_heatmap(year: int, month: int) -> pd.DataFrame:
    feature_df = make_monthly_features(year, month)

    if feature_df.empty:
        return pd.DataFrame(columns=["code", "dong", "gu", "grade", "score"])

    pred_df = predict_score(feature_df)
    pred_df = add_grade(pred_df)

    result = pred_df[["code", "dong", "gu", "grade", "score"]].copy()

    result["score"] = pd.to_numeric(result["score"], errors="coerce").fillna(0)
    result["score"] = result["score"].round(2)
    result["grade"] = result["grade"].astype(int)

    return result