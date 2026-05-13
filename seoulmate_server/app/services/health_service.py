from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "app/data/monthly_risk.csv"


def load_health_data() -> pd.DataFrame:
    """
    monthly_risk.csv를 읽어서 FastAPI heatmap 응답용 표준 컬럼으로 변환한다.

    원본:
    - 행정동_코드
    - 행정동_한글
    - 자치구명
    - 건강안전도_점수
    - 건강안전도_등급

    API:
    - code
    - dong
    - gu
    - score
    - grade
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"health 데이터 파일이 없습니다: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, dtype=str)
    df.columns = df.columns.str.strip()

    required_cols = [
        "행정동_코드",
        "행정동_한글",
        "자치구명",
        "year",
        "month",
        "건강안전도_점수",
        "건강안전도_등급",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"monthly_risk.csv 필수 컬럼 누락: {missing}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df["건강안전도_점수"] = pd.to_numeric(df["건강안전도_점수"], errors="coerce")
    df["건강안전도_등급"] = pd.to_numeric(df["건강안전도_등급"], errors="coerce")

    df = df.dropna(
        subset=[
            "행정동_코드",
            "행정동_한글",
            "자치구명",
            "year",
            "month",
            "건강안전도_점수",
            "건강안전도_등급",
        ]
    ).copy()

    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    # 원본은 "건강안전도"라서 높을수록 안전함.
    # API health layer는 "건강위험도"로 쓰기 위해 100 - 안전도 점수로 변환.
    df["score"] = (100 - df["건강안전도_점수"]).clip(0, 100).round(2)

    # 원본 건강안전도_등급은 1이 안전, 5가 위험한 쪽으로 이미 나뉘어 있음.
    # 그래서 health risk grade로 그대로 사용.
    df["grade"] = df["건강안전도_등급"].astype(int)

    result = df.rename(
        columns={
            "행정동_코드": "code",
            "행정동_한글": "dong",
            "자치구명": "gu",
        }
    )

    result["code"] = result["code"].astype(str).str.replace(".0", "", regex=False).str.strip()
    result["dong"] = result["dong"].astype(str).str.strip()
    result["gu"] = result["gu"].astype(str).str.strip()

    return result


def get_heatmap(year: int, month: int) -> pd.DataFrame:
    """
    건강위험도 heatmap 조회.

    현재 monthly_risk.csv는 룰베이스 결과 파일이므로,
    요청한 year/month가 파일에 존재해야 반환된다.
    """

    df = load_health_data()

    target = df[
        (df["year"] == int(year)) &
        (df["month"] == int(month))
    ].copy()

    if target.empty:
        print(f"[health] 데이터 없음: year={year}, month={month}")
        return pd.DataFrame(columns=["code", "dong", "gu", "grade", "score"])

    target = target[["code", "dong", "gu", "grade", "score"]].copy()

    # 혹시 같은 월/동 중복이 있으면 마지막 값 사용
    target = (
        target
        .drop_duplicates(subset=["code", "dong", "gu"], keep="last")
        .reset_index(drop=True)
    )

    return target