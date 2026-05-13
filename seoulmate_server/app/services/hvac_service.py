from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "app/models/hvac_model.pkl"
HVAC_STANDARD_PATH = BASE_DIR / "app/data/hvac_history_standard.csv"

_model_package = None


HVAC_GRADE_BINS = [-0.1, 20, 40, 60, 80, 100]
HVAC_GRADE_LABELS = [1, 2, 3, 4, 5]



def normalize_text(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace(".", "·", regex=False)
        .str.replace("ㆍ", "·", regex=False)
        .str.replace("・", "·", regex=False)
    )


def attach_code_from_standard(monthly: pd.DataFrame) -> pd.DataFrame:
    monthly = monthly.copy()

    # 이미 code가 있으면 그대로 사용
    if "code" in monthly.columns:
        monthly["code"] = monthly["code"].astype(str).str.strip()
        monthly["gu"] = monthly.get("gu", monthly["자치구"]).astype(str).str.strip()
        monthly["dong"] = monthly.get("dong", monthly["행정동"]).astype(str).str.strip()
        return monthly

    # =====================================================
    # 1. 모델 입력용 컬럼은 그대로 유지
    #    자치구 / 행정동은 LightGBM 입력 feature로 사용될 수 있음
    # =====================================================

    # =====================================================
    # 2. API 응답 및 code 매핑용 컬럼 생성
    # =====================================================
    monthly["out_gu"] = monthly["자치구"].astype(str).str.strip()
    monthly["out_dong"] = monthly["행정동"].astype(str).str.strip()

    # =====================================================
    # 3. 수동 보정
    # =====================================================
    manual_fix = {
        ("관악구", "삼성"): ("관악구", "삼성동"),
        ("강남구", "일원2동"): ("강남구", "일원본동"),
    }

    for (old_gu, old_dong), (new_gu, new_dong) in manual_fix.items():
        mask = (
            (monthly["out_gu"] == old_gu) &
            (monthly["out_dong"] == old_dong)
        )

        print(
            f"[hvac code mapping] 수동 보정 "
            f"{old_gu} {old_dong} -> {new_gu} {new_dong}: {int(mask.sum())} rows"
        )

        monthly.loc[mask, "out_gu"] = new_gu
        monthly.loc[mask, "out_dong"] = new_dong

    # =====================================================
    # 4. 전농3동 제거
    # =====================================================
    drop_mask = (
        (monthly["out_gu"] == "동대문구") &
        (monthly["out_dong"] == "전농3동")
    )

    print("[hvac code mapping] 전농3동 drop rows:", int(drop_mask.sum()))

    monthly = monthly.loc[~drop_mask].copy()

    # =====================================================
    # 5. 상일동은 상일1동 / 상일2동으로 복제
    #    단, 모델 입력용 행정동은 상일동 그대로 유지
    #    API 응답용 out_dong만 상일1동/상일2동으로 변경
    # =====================================================
    sangil_mask = (
        (monthly["out_gu"] == "강동구") &
        (monthly["out_dong"] == "상일동")
    )

    sangil_rows = monthly.loc[sangil_mask].copy()

    print("[hvac code mapping] 상일동 원본 rows:", len(sangil_rows))

    if len(sangil_rows) > 0:
        sangil_1 = sangil_rows.copy()
        sangil_2 = sangil_rows.copy()

        sangil_1["out_dong"] = "상일1동"
        sangil_2["out_dong"] = "상일2동"

        monthly = monthly.loc[~sangil_mask].copy()
        monthly = pd.concat([monthly, sangil_1, sangil_2], ignore_index=True)

        print(
            "[hvac code mapping] 상일동 복제 후 추가 rows:",
            len(sangil_1) + len(sangil_2)
        )

    # =====================================================
    # 6. hvac_history_standard.csv에서 code map 만들기
    # =====================================================
    standard = pd.read_csv(
        HVAC_STANDARD_PATH,
        dtype=str,
        usecols=["code", "gu", "dong"]
    )

    standard.columns = standard.columns.str.strip()

    code_map = (
        standard[["code", "gu", "dong"]]
        .dropna(subset=["code", "gu", "dong"])
        .drop_duplicates()
        .copy()
    )

    code_map["gu_key"] = normalize_text(code_map["gu"])
    code_map["dong_key"] = normalize_text(code_map["dong"])
    code_map["code"] = code_map["code"].astype(str).str.strip()

    code_map = (
        code_map[["gu_key", "dong_key", "code"]]
        .drop_duplicates(subset=["gu_key", "dong_key"], keep="first")
    )

    # monthly_history의 출력용 이름 기준으로 매칭
    monthly["gu_key"] = normalize_text(monthly["out_gu"])
    monthly["dong_key"] = normalize_text(monthly["out_dong"])

    merged = monthly.merge(
        code_map,
        on=["gu_key", "dong_key"],
        how="left"
    )

    fail = (
        merged[merged["code"].isna()][["out_gu", "out_dong"]]
        .drop_duplicates()
    )

    if len(fail) > 0:
        raise ValueError(
            "hvac monthly_history code 매핑 실패: "
            f"{fail.to_dict('records')[:30]}"
        )

    # =====================================================
    # 7. API 응답용 gu/dong 생성
    # =====================================================
    merged["gu"] = merged["out_gu"]
    merged["dong"] = merged["out_dong"]

    merged = merged.drop(
        columns=["gu_key", "dong_key", "out_gu", "out_dong"],
        errors="ignore"
    )

    return merged


def load_model_package():
    global _model_package

    if _model_package is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"hvac 모델 파일이 없음: {MODEL_PATH}")

        package = joblib.load(MODEL_PATH)

        if not isinstance(package, dict):
            raise ValueError("hvac_model.pkl은 dict 패키지 형태여야 함")

        required_keys = ["model", "feature_names", "monthly_history"]
        missing_keys = [k for k in required_keys if k not in package]

        if missing_keys:
            raise ValueError(
                f"hvac_model.pkl 필수 key 없음: {missing_keys}. "
                f"현재 keys: {list(package.keys())}"
            )

        if "cat_cols" not in package:
            package["cat_cols"] = []

        print("[hvac model loaded]")
        print("keys:", list(package.keys()))
        print("feature count:", len(package["feature_names"]))
        print("feature sample:", package["feature_names"][:20])
        print("cat_cols:", package.get("cat_cols"))
        print("target_type:", package.get("target_type"))

        _model_package = package

    return _model_package


def prepare_monthly_history() -> pd.DataFrame:
    package = load_model_package()

    history = package["monthly_history"]

    if not isinstance(history, pd.DataFrame):
        raise ValueError("package['monthly_history']가 DataFrame이 아님")

    df = history.copy()
    df.columns = df.columns.str.strip()

    required_cols = ["자치구", "행정동", "year", "month"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"monthly_history 필수 컬럼 없음: {missing}. "
            f"현재 컬럼: {df.columns.tolist()}"
        )

    # code는 hvac_history_standard.csv에서 붙임
    df = attach_code_from_standard(df)

    df["code"] = df["code"].astype(str).str.strip()

    # attach_code_from_standard()에서 만든 API 응답용 gu/dong을 유지해야 함
    df["gu"] = df["gu"].astype(str).str.strip()
    df["dong"] = df["dong"].astype(str).str.strip()

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype(int)

    if "quarter" not in df.columns:
        df["quarter"] = ((df["month"] - 1) // 3 + 1).astype(int)

    if "month_sin" not in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)

    if "month_cos" not in df.columns:
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def build_X(feature_df: pd.DataFrame) -> pd.DataFrame:
    package = load_model_package()

    feature_names = list(package["feature_names"])
    cat_cols = list(package.get("cat_cols", []))

    missing_features = [c for c in feature_names if c not in feature_df.columns]

    if missing_features:
        raise ValueError(
            f"hvac 모델 입력 피처가 monthly_history에 없음: {missing_features}. "
            f"현재 컬럼: {feature_df.columns.tolist()}"
        )

    X = feature_df.reindex(columns=feature_names).copy()

    for col in X.columns:
        if col in cat_cols:
            X[col] = X[col].astype("category")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    return X


# def add_grade(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()

#     if df.empty:
#         df["grade"] = []
#         return df

#     df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).clip(0, 100)

#     df["grade"] = pd.cut(
#         df["score"],
#         bins=HVAC_GRADE_BINS,
#         labels=HVAC_GRADE_LABELS,
#         include_lowest=True,
#     ).astype(int)

#     return df


def get_heatmap(year: int, month: int) -> pd.DataFrame:
    package = load_model_package()
    model = package["model"]

    history = prepare_monthly_history()

    target_date = pd.Timestamp(year=int(year), month=int(month), day=1)
    source_date = target_date - pd.DateOffset(months=1)

    source_year = source_date.year
    source_month = source_date.month

    source = history[
        (history["year"] == int(source_year)) &
        (history["month"] == int(source_month))
    ].copy()

    if source.empty:
        return pd.DataFrame(columns=["code", "dong", "gu", "grade", "score"])

    X = build_X(source)

    pred_absolute = np.clip(model.predict(X), 0, 100)

    result = source[["code", "dong", "gu"]].copy()
    result["absolute_score"] = pred_absolute

    # 히트맵용 score: 요청월 안에서 상대순위
    result["score"] = (
        result["absolute_score"]
        .rank(pct=True, method="average") * 100
    ).clip(0, 100)

    result["grade"] = pd.cut(
        result["score"],
        bins=[-0.1, 20, 40, 60, 80, 100],
        labels=[1, 2, 3, 4, 5],
        include_lowest=True,
    ).astype(int)

    result["score"] = result["score"].round(2)

    return result[["code", "dong", "gu", "grade", "score"]]



# def add_hvac_score_and_grade(
#     df: pd.DataFrame,
#     pred_col: str = "pred_absolute",
# ) -> pd.DataFrame:
#     """
#     HVAC 예측 결과 후처리.

#     score:
#         모델이 직접 예측한 냉난방 수요 점수, 0~100.

#     grade:
#         같은 year/month 안에서 상대순위 기준 5등급.
#         1 = 상대적으로 낮음
#         5 = 상대적으로 높음
#     """
#     result = df.copy()

#     if pred_col not in result.columns:
#         raise KeyError(
#             f"예측 컬럼이 없습니다: {pred_col}, 현재 컬럼: {result.columns.tolist()}"
#         )

#     result[pred_col] = pd.to_numeric(result[pred_col], errors="coerce").clip(0, 100)

#     if result[pred_col].isna().any():
#         bad_count = int(result[pred_col].isna().sum())
#         raise ValueError(f"HVAC 예측값 NaN 발생: {bad_count} rows")

#     # =====================================================
#     # 1. API score는 모델 예측 점수 그대로 사용
#     # =====================================================
#     result["score"] = result[pred_col].round(2)

#     # 디버깅용으로 원본 예측값도 보존
#     result["absolute_score"] = result[pred_col].round(2)

#     # =====================================================
#     # 2. grade는 월별 상대순위 기준으로 생성
#     # =====================================================
#     if "year" in result.columns and "month" in result.columns:
#         group_cols = ["year", "month"]
#     elif "year_month" in result.columns:
#         group_cols = ["year_month"]
#     else:
#         group_cols = None

#     if group_cols:
#         result["relative_score"] = (
#             result.groupby(group_cols)[pred_col]
#             .rank(method="average", pct=True)
#             * 100
#         )
#     else:
#         result["relative_score"] = (
#             result[pred_col].rank(method="average", pct=True) * 100
#         )

#     result["relative_score"] = result["relative_score"].round(2)

#     # =====================================================
#     # 3. 상대점수 기준 5등급
#     # 0~20   -> 1
#     # 20~40  -> 2
#     # 40~60  -> 3
#     # 60~80  -> 4
#     # 80~100 -> 5
#     # =====================================================
#     result["grade"] = np.ceil(result["relative_score"] / 20).astype(int)
#     result["grade"] = result["grade"].clip(1, 5)

#     return result