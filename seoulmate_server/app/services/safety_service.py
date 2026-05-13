from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

SAFETY_PATH = BASE_DIR / "app/data/safety_model.csv"
DONG_MAP_PATH = BASE_DIR / "app/data/행정동_with_코드.csv"

_df = None


def normalize_text(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace(".", "·", regex=False)
        .str.replace("ㆍ", "·", regex=False)
        .str.replace("・", "·", regex=False)
    )


def load_dong_map() -> pd.DataFrame:
    """
    행정동_with_코드.csv에서
    자치구, 행정동_표준명, 행정동코드를 가져온다.
    """

    if not DONG_MAP_PATH.exists():
        raise FileNotFoundError(f"행정동 매핑 파일이 없습니다: {DONG_MAP_PATH}")

    dong_map = pd.read_csv(DONG_MAP_PATH, dtype=str)
    dong_map.columns = dong_map.columns.str.strip()

    required_cols = ["자치구", "행정동_표준명", "행정동코드"]
    missing = [c for c in required_cols if c not in dong_map.columns]

    if missing:
        raise ValueError(
            f"행정동_with_코드.csv 필수 컬럼 누락: {missing}, "
            f"현재 컬럼: {dong_map.columns.tolist()}"
        )

    dong_map = dong_map[required_cols].copy()

    dong_map = dong_map.rename(
        columns={
            "자치구": "gu",
            "행정동_표준명": "dong",
            "행정동코드": "code",
        }
    )

    dong_map["gu"] = dong_map["gu"].astype(str).str.strip()
    dong_map["dong"] = dong_map["dong"].astype(str).str.strip()
    dong_map["code"] = (
        dong_map["code"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )

    dong_map["gu_key"] = normalize_text(dong_map["gu"])
    dong_map["dong_key"] = normalize_text(dong_map["dong"])

    # 같은 행정동코드 중복 제거
    dong_map = (
        dong_map[["code", "gu", "dong", "gu_key", "dong_key"]]
        .drop_duplicates(subset=["code"], keep="last")
        .reset_index(drop=True)
    )

    return dong_map


def score_to_grade(score: float) -> int:
    """
    score 기준 5등급.
    여기서는 safety_model.csv의 등급이 없거나 결측일 때만 보조로 사용.
    """
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


def load_data() -> pd.DataFrame:
    """
    safety_model.csv는 자치구 단위 데이터다.

    원본 구조:
    - 자치구
    - 2017_점수, 2017_등급
    - ...
    - 2024_점수, 2024_등급

    API 구조:
    - code
    - dong
    - gu
    - grade
    - score

    처리 방식:
    자치구별 2024_점수/등급을 해당 자치구의 모든 행정동에 복제한다.
    """

    global _df

    if _df is not None:
        return _df

    if not SAFETY_PATH.exists():
        raise FileNotFoundError(f"safety 데이터 파일이 없습니다: {SAFETY_PATH}")

    safety = pd.read_csv(SAFETY_PATH, dtype=str)
    safety.columns = safety.columns.str.strip()

    print("[safety columns]", safety.columns.tolist())

    required_cols = ["자치구", "2024_점수", "2024_등급"]
    missing = [c for c in required_cols if c not in safety.columns]

    if missing:
        raise ValueError(
            f"safety_model.csv 필수 컬럼 누락: {missing}, "
            f"현재 컬럼: {safety.columns.tolist()}"
        )

    safety = safety[["자치구", "2024_점수", "2024_등급"]].copy()

    safety = safety.rename(
        columns={
            "자치구": "gu",
            "2024_점수": "score",
            "2024_등급": "grade",
        }
    )

    safety["gu"] = safety["gu"].astype(str).str.strip()
    safety["gu_key"] = normalize_text(safety["gu"])

    safety["score"] = pd.to_numeric(safety["score"], errors="coerce")
    safety["grade"] = pd.to_numeric(safety["grade"], errors="coerce")

    safety = safety.dropna(subset=["gu", "score"]).copy()

    safety["score"] = safety["score"].clip(0, 100).round(2)

    missing_grade = safety["grade"].isna()
    if missing_grade.any():
        safety.loc[missing_grade, "grade"] = safety.loc[missing_grade, "score"].apply(score_to_grade)

    safety["grade"] = safety["grade"].astype(int)

    dong_map = load_dong_map()

    # 자치구 점수를 행정동 전체에 복제
    result = dong_map.merge(
        safety[["gu_key", "score", "grade"]],
        on="gu_key",
        how="left",
    )

    # safety 점수가 없는 자치구 확인
    missing_safety = result[result["score"].isna()][["gu"]].drop_duplicates()

    if not missing_safety.empty:
        print("[safety] 점수 매칭 실패 자치구:")
        print(missing_safety.to_string(index=False))

    result = result.dropna(subset=["score", "grade"]).copy()

    result["grade"] = result["grade"].astype(int)
    result["score"] = result["score"].astype(float)

    result = (
        result[["code", "dong", "gu", "grade", "score"]]
        .drop_duplicates(subset=["code"], keep="last")
        .reset_index(drop=True)
    )

    _df = result

    return _df


def get_heatmap(year: int, month: int) -> pd.DataFrame:
    """
    치안 점수는 현재 2024년 기준 자치구 룰베이스 고정 점수다.

    따라서 year/month 요청값은 받지만,
    safety_model.csv에 월별 데이터가 없으므로 모든 요청 월에 같은 값을 반환한다.
    """

    df = load_data().copy()

    if df.empty:
        print(f"[safety] 데이터 없음: year={year}, month={month}")
        return pd.DataFrame(columns=["code", "dong", "gu", "grade", "score"])

    return df[["code", "dong", "gu", "grade", "score"]].copy()