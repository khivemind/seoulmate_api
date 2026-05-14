from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

OLD_RAW_PATH = BASE_DIR / "app/data/stress_history.csv"
NEW_RAW_PATH = BASE_DIR / "app/data/4월_센서데이터.csv"

BACKUP_PATH = BASE_DIR / "app/data/stress_history_update.csv"
OUTPUT_PATH = BASE_DIR / "app/data/stress_history.csv"


def read_csv_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="cp949")


def parse_datetime_safely(df: pd.DataFrame) -> pd.Series:
    """
    측정시간을 최대한 안전하게 datetime으로 변환.
    1차: 일반 문자열 datetime
    2차: 숫자형 Excel serial date
    3차: 등록일시 fallback
    """
    if "측정시간" not in df.columns:
        raise ValueError(f"측정시간 컬럼이 없음. 현재 컬럼: {df.columns.tolist()}")

    s = df["측정시간"].astype(str).str.strip()
    s = s.str.replace("_", " ", regex=False)

    # 빈 문자열 처리
    s = s.replace({"": None, "nan": None, "NaN": None, "None": None})

    # 1차: 일반 날짜 문자열 파싱
    dt = pd.to_datetime(
        s,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    # 2차: Excel serial number 형태 대응
    # 예: 45382.123 같은 값
    num = pd.to_numeric(s, errors="coerce")
    excel_dt = pd.to_datetime(
        num,
        unit="D",
        origin="1899-12-30",
        errors="coerce",
    )

    dt = dt.fillna(excel_dt)

    # 3차: 등록일시 fallback
    if "등록일시" in df.columns:
        reg = df["등록일시"].astype(str).str.strip()
        reg = reg.replace({"": None, "nan": None, "NaN": None, "None": None})
        reg_dt = pd.to_datetime(reg, errors="coerce")
        dt = dt.fillna(reg_dt)

    return dt


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    dt = parse_datetime_safely(df)

    bad = int(dt.isna().sum())
    print("측정시간 변환 실패 rows:", bad)

    if bad > 0:
        print("\n측정시간 변환 실패 샘플:")
        print(df.loc[dt.isna(), ["측정시간"]].head(20).to_string(index=False))

        if "등록일시" in df.columns:
            print("\n등록일시 샘플:")
            print(df.loc[dt.isna(), ["등록일시"]].head(20).to_string(index=False))

    # 실패가 너무 많으면 중단
    if bad > 100:
        raise ValueError(
            f"측정시간 변환 실패가 너무 많음: {bad} rows. "
            "4월_센서데이터.csv의 측정시간 형식을 먼저 확인해야 함."
        )

    # 표준 문자열로 다시 저장
    df["측정시간"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S")

    df["month"] = dt.dt.month.astype("Int64").astype(str)
    df["hour"] = dt.dt.hour.astype("Int64").astype(str)
    df["day"] = dt.dt.day.astype("Int64").astype(str)
    df["dayofweek"] = dt.dt.dayofweek.astype("Int64").astype(str)
    df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int).astype(str)

    return df


def main():
    print("기존 stress_history.csv 읽는 중...")
    old_df = read_csv_safely(OLD_RAW_PATH)

    print("새 4월 한글 변환 데이터 읽는 중...")
    new_df = read_csv_safely(NEW_RAW_PATH)

    old_df.columns = old_df.columns.str.strip()
    new_df.columns = new_df.columns.str.strip()

    print("기존 shape:", old_df.shape)
    print("새 데이터 shape:", new_df.shape)

    # 백업
    old_df.to_csv(BACKUP_PATH, index=False, encoding="utf-8-sig")
    print("백업 저장:", BACKUP_PATH)

    # 새 데이터에 시간 파생 컬럼 추가
    new_df = add_time_features(new_df)

    # old_df 컬럼 기준으로 new_df 맞추기
    old_cols = old_df.columns.tolist()

    missing_in_new = [c for c in old_cols if c not in new_df.columns]
    if missing_in_new:
        raise ValueError(
            f"새 4월 데이터에 기존 stress_history 컬럼이 부족함: {missing_in_new}"
        )

    new_part = new_df[old_cols].copy()
    old_part = old_df[old_cols].copy()

    print("\nold_part shape:", old_part.shape)
    print("new_part shape:", new_part.shape)

    combined = pd.concat([old_part, new_part], ignore_index=True)

    print("\n합친 후 shape:", combined.shape)

    # 매우 중요:
    # 자치구+행정동+측정시간 기준 중복 제거 금지
    # 센서 여러 개가 같은 시간/동에 존재할 수 있음
    before = len(combined)
    combined = combined.drop_duplicates(keep="last")
    print("전체 행 완전 동일 기준 중복 제거 rows:", before - len(combined))

    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n저장 완료:", OUTPUT_PATH)
    print("최종 shape:", combined.shape)


if __name__ == "__main__":
    main()