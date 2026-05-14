from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "app/data"
OUT_DIR = DATA_DIR / "api"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMFORT_PATH = DATA_DIR / "comfort.csv"
HEALTH_PATH = DATA_DIR / "monthly_risk.csv"
SAFETY_PATH = DATA_DIR / "safety_model.csv"
DONG_MAP_PATH = DATA_DIR / "행정동_with_코드.csv"


def normalize_code(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )


def clean_text(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace(".", "·", regex=False)
        .str.replace("ㆍ", "·", regex=False)
        .str.replace("・", "·", regex=False)
    )


def save_layer_month_files(df: pd.DataFrame, layer: str, year_col: str, month_col: str):
    """
    표준 컬럼을 가진 df를 월별 API 결과 파일로 저장한다.

    필요 컬럼:
    code, dong, gu, grade, score, year_col, month_col
    """

    required = ["code", "dong", "gu", "grade", "score", year_col, month_col]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"[{layer}] 필수 컬럼 누락: {missing}")

    df = df.copy()

    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df[month_col] = pd.to_numeric(df[month_col], errors="coerce")

    df = df.dropna(subset=[year_col, month_col, "code", "dong", "gu", "score", "grade"]).copy()

    df[year_col] = df[year_col].astype(int)
    df[month_col] = df[month_col].astype(int)

    df["code"] = normalize_code(df["code"])
    df["dong"] = df["dong"].astype(str).str.strip()
    df["gu"] = df["gu"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce").round(2)
    df["grade"] = pd.to_numeric(df["grade"], errors="coerce").astype(int)

    for (year, month), part in df.groupby([year_col, month_col]):
        year = int(year)
        month = int(month)

        out = part[["code", "dong", "gu", "grade", "score"]].copy()

        out_path = OUT_DIR / f"{layer}_{year}_{month:02d}.csv"
        out.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"[{layer}] 저장 완료: {out_path.name} shape={out.shape}")


def build_comfort_api_results():
    """
    comfort.csv:
    - 행정동코드
    - 자치구
    - 행정동
    - 년도
    - 월
    - 쾌적도점수
    - 쾌적도등급

    이미 계산된 쾌적도점수/등급을 그대로 사용한다.
    """

    print("\n[comfort] 원본 읽는 중:", COMFORT_PATH)

    df = pd.read_csv(COMFORT_PATH, dtype=str)
    df.columns = df.columns.str.strip()

    required = ["행정동코드", "자치구", "행정동", "년도", "월", "쾌적도점수", "쾌적도등급"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"comfort.csv 필수 컬럼 누락: {missing}, 현재 컬럼: {df.columns.tolist()}")

    out = pd.DataFrame({
        "code": df["행정동코드"],
        "dong": df["행정동"],
        "gu": df["자치구"],
        "year": df["년도"],
        "month": df["월"],
        "score": df["쾌적도점수"],
        "grade": df["쾌적도등급"],
    })

    save_layer_month_files(out, "comfort", "year", "month")


def build_health_api_results():
    """
    monthly_risk.csv:
    - 행정동_코드
    - 행정동_한글
    - 자치구명
    - year
    - month
    - 건강안전도_점수
    - 건강안전도_등급

    이미 계산된 건강안전도_점수/등급을 그대로 사용한다.
    """

    print("\n[health] 원본 읽는 중:", HEALTH_PATH)

    df = pd.read_csv(HEALTH_PATH, dtype=str)
    df.columns = df.columns.str.strip()

    required = ["행정동_코드", "행정동_한글", "자치구명", "year", "month", "건강안전도_점수", "건강안전도_등급"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"monthly_risk.csv 필수 컬럼 누락: {missing}, 현재 컬럼: {df.columns.tolist()}")

    out = pd.DataFrame({
        "code": df["행정동_코드"],
        "dong": df["행정동_한글"],
        "gu": df["자치구명"],
        "year": df["year"],
        "month": df["month"],
        "score": df["건강안전도_점수"],
        "grade": df["건강안전도_등급"],
    })

    save_layer_month_files(out, "health", "year", "month")


def load_dong_master() -> pd.DataFrame:
    """
    safety는 자치구 단위라서 행정동 단위로 확장하기 위해
    행정동_with_코드.csv를 사용한다.
    """

    print("\n[dong master] 행정동 매핑 읽는 중:", DONG_MAP_PATH)

    m = pd.read_csv(DONG_MAP_PATH, dtype=str)
    m.columns = m.columns.str.strip()

    required = ["자치구", "행정동_표준명", "행정동코드"]
    missing = [col for col in required if col not in m.columns]

    if missing:
        raise ValueError(f"행정동_with_코드.csv 필수 컬럼 누락: {missing}, 현재 컬럼: {m.columns.tolist()}")

    master = pd.DataFrame({
        "gu": m["자치구"],
        "dong": m["행정동_표준명"],
        "code": m["행정동코드"],
    })

    master["gu"] = master["gu"].astype(str).str.strip()
    master["dong"] = master["dong"].astype(str).str.strip()
    master["code"] = normalize_code(master["code"])

    master = master.dropna(subset=["gu", "dong", "code"]).copy()
    master = master.drop_duplicates(subset=["code", "gu", "dong"]).copy()

    # 서울대공원 같은 분석 제외 대상이 섞여 있으면 제거
    master = master[master["gu"] != "서울대공원"].copy()

    print("[dong master] shape:", master.shape)
    print(master.head())

    return master



def copy_latest_safety_to_future_years(latest_year: int, future_years: list[int]):
    """
    safety_model.csv의 최신 연도 데이터를
    미래 연도 파일로 복사한다.

    예:
    safety_2024_01.csv -> safety_2025_01.csv
    safety_2024_01.csv -> safety_2026_01.csv
    """

    for future_year in future_years:
        for month in range(1, 13):
            src = OUT_DIR / f"safety_{latest_year}_{month:02d}.csv"
            dst = OUT_DIR / f"safety_{future_year}_{month:02d}.csv"

            if not src.exists():
                print(f"[safety copy] 원본 파일 없음: {src.name}")
                continue

            df = pd.read_csv(src, dtype=str)
            df.to_csv(dst, index=False, encoding="utf-8-sig")

        print(f"[safety copy] {latest_year}년 값을 {future_year}년 1~12월로 복사 완료")



def build_safety_api_results():
    """
    safety_model.csv:
    - 자치구
    - 2017_점수, 2017_등급
    - ...
    - 2024_점수, 2024_등급

    safety는 자치구 단위 점수이므로,
    해당 자치구의 모든 행정동에 같은 safety 점수/등급을 부여한다.

    safety는 월별 데이터가 아니라 연도별 데이터이므로,
    API 규격에 맞추기 위해 해당 연도의 1~12월 파일을 모두 생성한다.
    """

    print("\n[safety] 원본 읽는 중:", SAFETY_PATH)

    safety = pd.read_csv(SAFETY_PATH, dtype=str)
    safety.columns = safety.columns.str.strip()

    if "자치구" not in safety.columns:
        raise ValueError(f"safety_model.csv에 자치구 컬럼이 없음. 현재 컬럼: {safety.columns.tolist()}")

    dong_master = load_dong_master()

    available_years = []

    for col in safety.columns:
        if col.endswith("_점수"):
            year = col.replace("_점수", "")
            grade_col = f"{year}_등급"

            if grade_col in safety.columns and year.isdigit():
                available_years.append(int(year))

    available_years = sorted(available_years)

    if not available_years:
        raise ValueError("safety_model.csv에서 사용 가능한 연도별 점수/등급 컬럼을 찾지 못함")

    print("[safety] 사용 가능한 연도:", available_years)

    safety = safety.rename(columns={"자치구": "gu"})
    safety["gu"] = safety["gu"].astype(str).str.strip()

    for year in available_years:
        score_col = f"{year}_점수"
        grade_col = f"{year}_등급"

        temp = safety[["gu", score_col, grade_col]].copy()
        temp = temp.rename(columns={
            score_col: "score",
            grade_col: "grade",
        })

        merged = dong_master.merge(temp, on="gu", how="left")

        missing_rows = merged["score"].isna().sum()
        if missing_rows > 0:
            print(f"[safety {year}] 자치구 매칭 실패 rows: {missing_rows}")

        merged = merged.dropna(subset=["score", "grade"]).copy()

        merged["score"] = pd.to_numeric(merged["score"], errors="coerce").round(2)
        merged["grade"] = pd.to_numeric(merged["grade"], errors="coerce").astype(int)

        out = merged[["code", "dong", "gu", "grade", "score"]].copy()

        # safety는 연도별 데이터라서 1~12월 같은 값으로 저장
        for month in range(1, 13):
            out_path = OUT_DIR / f"safety_{year}_{month:02d}.csv"
            out.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"[safety] {year}년 1~12월 저장 완료 shape={out.shape}")

    latest_year = max(available_years)
    copy_latest_safety_to_future_years(latest_year, [2025, 2026])


def main():
    build_comfort_api_results()
    build_health_api_results()
    build_safety_api_results()

    print("\n전체 룰베이스 API 결과 생성 완료")


if __name__ == "__main__":
    main()