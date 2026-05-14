from pathlib import Path

import numpy as np
import pandas as pd


# 대용량 파일 위치
# 서버 밖으로 빼면 이 경로만 바꿔주면 됨
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = Path("/Users/hayden_rim/Desktop/seoulmate_data/raw/stress_history_standard.csv")
OUT_DIR = BASE_DIR / "app/data/api"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def score_to_grade(score: float) -> int:
    if score <= 20:
        return 1
    elif score <= 40:
        return 2
    elif score <= 60:
        return 3
    elif score <= 80:
        return 4
    else:
        return 5


def build_monthly_stress(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "소음 평균(dB)",
        "소음 최대(dB)",
        "소음 최소(dB)",
        "진동(x) 평균(mm/s)",
        "진동(y) 평균(mm/s)",
        "진동(z) 평균(mm/s)",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    monthly = (
        df.groupby(["code", "gu", "dong", "year", "month"], as_index=False)
        .agg(
            noise_avg=("소음 평균(dB)", "mean"),
            noise_max=("소음 최대(dB)", "mean"),
            noise_min=("소음 최소(dB)", "mean"),
            vib_x=("진동(x) 평균(mm/s)", "mean"),
            vib_y=("진동(y) 평균(mm/s)", "mean"),
            vib_z=("진동(z) 평균(mm/s)", "mean"),
            n_obs=("소음 평균(dB)", "count"),
        )
    )

    monthly["vibration_magnitude"] = np.sqrt(
        monthly["vib_x"].fillna(0) ** 2
        + monthly["vib_y"].fillna(0) ** 2
        + monthly["vib_z"].fillna(0) ** 2
    )

    # 소음 점수: 60dB 기준 초과분 반영
    monthly["noise_score"] = ((monthly["noise_avg"] - 45) / 35 * 100).clip(0, 100)

    # 진동 점수: 월별 상대 순위 기반
    monthly["vibration_score"] = (
        monthly.groupby(["year", "month"])["vibration_magnitude"]
        .rank(pct=True, method="average")
        * 100
    )

    # 최종 stress score
    monthly["stress_raw_score"] = (
        monthly["noise_score"] * 0.7
        + monthly["vibration_score"] * 0.3
    ).clip(0, 100)
    
    monthly["score"] = (
        monthly.groupby(["year", "month"])["stress_raw_score"]
        .rank(pct=True, method="average")
        * 100
    ).round(2)
    
    monthly["grade"] = pd.cut(
        monthly["score"],
        bins=[-0.1, 20, 40, 60, 80, 100],
        labels=[1, 2, 3, 4, 5],
        include_lowest=True,
    ).astype(int)

    return monthly[["code", "dong", "gu", "grade", "score", "year", "month"]]


def save_month_files(monthly: pd.DataFrame):
    for (year, month), part in monthly.groupby(["year", "month"]):
        year = int(year)
        month = int(month)

        out = part[["code", "dong", "gu", "grade", "score"]].copy()

        out_path = OUT_DIR / f"stress_{year}_{month:02d}.csv"
        out.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"저장: {out_path} shape={out.shape}")


def main():
    print("stress_history_standard.csv 읽는 중...")
    df = pd.read_csv(INPUT_PATH, dtype=str)

    df.columns = df.columns.str.strip()

    df["code"] = df["code"].astype(str).str.replace(".0", "", regex=False).str.strip()
    df["gu"] = df["gu"].astype(str).str.strip()
    df["dong"] = df["dong"].astype(str).str.strip()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")

    df = df.dropna(subset=["code", "gu", "dong", "year", "month"]).copy()

    print("월별 stress score 생성 중...")
    monthly = build_monthly_stress(df)

    print("monthly shape:", monthly.shape)
    print(monthly.head())

    save_month_files(monthly)

    print("완료")


if __name__ == "__main__":
    main()