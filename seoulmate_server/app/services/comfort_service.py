from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
COMFORT_PATH = BASE_DIR / "app/data/comfort.csv"

_comfort_df = None


def load_comfort_data() -> pd.DataFrame:
    global _comfort_df

    if _comfort_df is None:
        if not COMFORT_PATH.exists():
            raise FileNotFoundError(f"comfort.csv 파일이 없음: {COMFORT_PATH}")

        df = pd.read_csv(COMFORT_PATH, dtype=str)
        df.columns = df.columns.str.strip()

        required_cols = [
            "행정동코드",
            "년도",
            "월",
            "자치구",
            "행정동",
            "쾌적도점수",
            "쾌적도등급",
        ]

        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            raise ValueError(
                f"comfort.csv 필수 컬럼 없음: {missing}. "
                f"현재 컬럼: {df.columns.tolist()}"
            )

        # 표준 컬럼 생성
        df["code"] = df["행정동코드"].astype(str).str.strip()
        df["gu"] = df["자치구"].astype(str).str.strip()
        df["dong"] = df["행정동"].astype(str).str.strip()

        df["year"] = pd.to_numeric(df["년도"], errors="coerce")
        df["month"] = pd.to_numeric(df["월"], errors="coerce")

        df["score"] = pd.to_numeric(df["쾌적도점수"], errors="coerce")
        df["grade"] = pd.to_numeric(df["쾌적도등급"], errors="coerce")

        # 결측 제거
        df = df.dropna(
            subset=["code", "gu", "dong", "year", "month", "score", "grade"]
        ).copy()

        df["year"] = df["year"].astype(int)
        df["month"] = df["month"].astype(int)
        df["grade"] = df["grade"].astype(int)

        # score 범위 방어
        df["score"] = df["score"].clip(0, 100)

        # grade 범위 방어
        df["grade"] = df["grade"].clip(1, 5)

        # 행정동코드 10자리만 사용
        invalid_code = df[df["code"].str.len() != 10]

        if len(invalid_code) > 0:
            raise ValueError(
                "comfort.csv에 10자리가 아닌 행정동코드가 있음: "
                f"{invalid_code[['code', 'gu', 'dong']].head(20).to_dict('records')}"
            )

        _comfort_df = df

    return _comfort_df


def get_heatmap(year: int, month: int) -> pd.DataFrame:
    df = load_comfort_data()

    target = df[
        (df["year"] == int(year)) &
        (df["month"] == int(month))
    ].copy()

    if target.empty:
        return pd.DataFrame(columns=["code", "dong", "gu", "grade", "score"])

    result = target[["code", "dong", "gu", "grade", "score"]].copy()

    result["score"] = pd.to_numeric(result["score"], errors="coerce").fillna(0)
    result["score"] = result["score"].round(2)

    result["grade"] = pd.to_numeric(result["grade"], errors="coerce").fillna(5)
    result["grade"] = result["grade"].astype(int).clip(1, 5)

    # 혹시 같은 행정동코드가 중복되면 하나만 남김
    result = (
        result.sort_values("score", ascending=False)
        .drop_duplicates(subset=["code"], keep="first")
        .copy()
    )

    return result