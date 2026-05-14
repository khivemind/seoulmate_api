from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

OLD_RAW_PATH = BASE_DIR / "app/data/hvac_history.csv"
NEW_RAW_PATH = BASE_DIR / "app/data/4월_센서데이터.csv"

BACKUP_PATH = BASE_DIR / "app/data/hvac_history_update.csv"
OUTPUT_PATH = BASE_DIR / "app/data/hvac_history.csv"


def read_csv_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="cp949")


def parse_datetime_safely(df: pd.DataFrame) -> pd.Series:
    if "측정시간" not in df.columns:
        raise ValueError(f"측정시간 컬럼이 없음. 현재 컬럼: {df.columns.tolist()}")

    s = df["측정시간"].astype(str).str.strip()
    s = s.str.replace("_", " ", regex=False)
    s = s.replace({"": None, "nan": None, "NaN": None, "None": None})

    dt = pd.to_datetime(
        s,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    if "등록일시" in df.columns:
        reg = df["등록일시"].astype(str).str.strip()
        reg = reg.str.replace("_", " ", regex=False)
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
        raise ValueError(f"측정시간 변환 실패가 있음: {bad} rows")

    df["측정시간"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["month"] = dt.dt.month.astype("Int64").astype(str)
    df["hour"] = dt.dt.hour.astype("Int64").astype(str)
    df["day"] = dt.dt.day.astype("Int64").astype(str)
    df["dayofweek"] = dt.dt.dayofweek.astype("Int64").astype(str)
    df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int).astype(str)

    return df


def main():
    print("기존 hvac_history.csv 읽는 중...")
    old_df = read_csv_safely(OLD_RAW_PATH)

    print("새 4월 한글 변환 데이터 읽는 중...")
    new_df = read_csv_safely(NEW_RAW_PATH)

    old_df.columns = old_df.columns.str.strip()
    new_df.columns = new_df.columns.str.strip()

    print("기존 shape:", old_df.shape)
    print("새 데이터 shape:", new_df.shape)

    old_df.to_csv(BACKUP_PATH, index=False, encoding="utf-8-sig")
    print("백업 저장:", BACKUP_PATH)

    new_df = add_time_features(new_df)

    old_cols = old_df.columns.tolist()

    missing_in_new = [c for c in old_cols if c not in new_df.columns]
    if missing_in_new:
        raise ValueError(
            f"새 4월 데이터에 기존 hvac_history 컬럼이 부족함: {missing_in_new}"
        )

    old_part = old_df[old_cols].copy()
    new_part = new_df[old_cols].copy()

    print("\nold_part shape:", old_part.shape)
    print("new_part shape:", new_part.shape)

    combined = pd.concat([old_part, new_part], ignore_index=True)

    print("\n합친 후 shape:", combined.shape)

    before = len(combined)
    combined = combined.drop_duplicates(keep="last")
    print("전체 행 완전 동일 기준 중복 제거 rows:", before - len(combined))

    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n저장 완료:", OUTPUT_PATH)
    print("최종 shape:", combined.shape)


if __name__ == "__main__":
    main()