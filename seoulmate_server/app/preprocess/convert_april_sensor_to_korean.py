from pathlib import Path

from normalize_new_sensor_regions import read_csv_safely, normalize_regions_with_dong_csv


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_PATH = BASE_DIR / "app/data/4월_센서데이터.csv"
DONG_MAP_PATH = BASE_DIR / "app/data/행정동.csv"

OUTPUT_PATH = BASE_DIR / "app/data/4월_센서데이터.csv"
FAILED_PATH = BASE_DIR / "app/data/4월_센서데이터_region_failed.csv"


def main():
    print("새 4월 센서 데이터 읽는 중...")
    raw_df = read_csv_safely(RAW_PATH)

    print("행정동.csv 읽는 중...")
    dong_map_df = read_csv_safely(DONG_MAP_PATH)

    print("\n[raw columns]")
    print(raw_df.columns.tolist())

    print("\n[dong map columns]")
    print(dong_map_df.columns.tolist())

    converted_df, failed_df = normalize_regions_with_dong_csv(
        raw_df=raw_df,
        dong_map_df=dong_map_df,
    )

    converted_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n변환 완료: {OUTPUT_PATH}")
    print("shape:", converted_df.shape)

    if len(failed_df) > 0:
        failed_df.to_csv(FAILED_PATH, index=False, encoding="utf-8-sig")
        print(f"\n매핑 실패 저장: {FAILED_PATH}")
        print(failed_df.head(30).to_string(index=False))
    else:
        print("\n매핑 실패 없음")


if __name__ == "__main__":
    main()