from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = Path("/Users/hayden_rim/Desktop/seoulmate_data/hvac_history_standard.csv")
OUT_DIR = BASE_DIR / "app/data/api"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def grade_by_month_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    월별 상대순위 score 0~100,
    grade는 5등급 균등 분포에 가깝게 나눔.
    """

    df["score"] = (
        df.groupby(["year", "month"])["냉난방_수요지수_raw"]
        .rank(pct=True, method="average")
        * 100
    ).round(2)

    df["grade"] = pd.cut(
        df["score"],
        bins=[-0.1, 20, 40, 60, 80, 100],
        labels=[1, 2, 3, 4, 5],
        include_lowest=True,
    ).astype(int)

    return df


def build_monthly_hvac(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "온도 평균(℃)",
        "온도 최대(℃)",
        "온도 최소(℃)",
        "습도 평균(%)",
        "조도 평균(lux)",
        "is_weekend",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    monthly = (
        df.groupby(["code", "gu", "dong", "year", "month"], as_index=False)
        .agg(
            temp_avg=("온도 평균(℃)", "mean"),
            temp_max=("온도 최대(℃)", "mean"),
            temp_min=("온도 최소(℃)", "mean"),
            humidity_avg=("습도 평균(%)", "mean"),
            light_avg=("조도 평균(lux)", "mean"),
            is_weekend=("is_weekend", "mean"),
            n_obs=("온도 평균(℃)", "count"),
        )
    )

    monthly["일교차"] = monthly["temp_max"] - monthly["temp_min"]

    # 불쾌지수 근사 공식
    monthly["불쾌지수"] = (
        0.81 * monthly["temp_avg"]
        + 0.01 * monthly["humidity_avg"] * (0.99 * monthly["temp_avg"] - 14.3)
        + 46.3
    )

    # 냉방 필요도: 더울수록 증가
    monthly["냉방_필요도"] = (monthly["temp_avg"] - 24).clip(lower=0)

    # 난방 필요도: 추울수록 증가
    monthly["난방_필요도"] = (18 - monthly["temp_avg"]).clip(lower=0)

    # 노트북의 raw 공식과 맞춰야 하는 부분
    monthly["냉난방_수요지수_raw"] = (
        monthly["냉방_필요도"] * 2.0
        + monthly["난방_필요도"] * 2.0
        + monthly["일교차"].clip(lower=0) * 0.3
        + (monthly["불쾌지수"] - 68).clip(lower=0) * 0.5
    )

    monthly = grade_by_month_rank(monthly)

    return monthly[["code", "dong", "gu", "grade", "score", "year", "month"]]


def save_month_files(monthly: pd.DataFrame):
    for (year, month), part in monthly.groupby(["year", "month"]):
        year = int(year)
        month = int(month)

        out = part[["code", "dong", "gu", "grade", "score"]].copy()

        out_path = OUT_DIR / f"hvac_{year}_{month:02d}.csv"
        out.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"저장: {out_path} shape={out.shape}")


def main():
    print("hvac_history_standard.csv 읽는 중...")
    df = pd.read_csv(INPUT_PATH, dtype=str)

    df.columns = df.columns.str.strip()

    df["code"] = df["code"].astype(str).str.replace(".0", "", regex=False).str.strip()
    df["gu"] = df["gu"].astype(str).str.strip()
    df["dong"] = df["dong"].astype(str).str.strip()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")

    df = df.dropna(subset=["code", "gu", "dong", "year", "month"]).copy()

    print("월별 hvac score 생성 중...")
    monthly = build_monthly_hvac(df)

    print("monthly shape:", monthly.shape)
    print(monthly.head())

    save_month_files(monthly)

    print("완료")


if __name__ == "__main__":
    main()