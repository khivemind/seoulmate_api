from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_PATH = BASE_DIR / "app/data/hvac_history.csv"
MAP_PATH = BASE_DIR / "app/data/행정동_with_코드.csv"
OUTPUT_PATH = BASE_DIR / "app/data/hvac_history_standard.csv"


def normalize_text(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace(".", "·", regex=False)
        .str.replace("ㆍ", "·", regex=False)
        .str.replace("・", "·", regex=False)
    )


def main():
    print("원본 hvac history 읽는 중...")
    hvac = pd.read_csv(RAW_PATH, dtype=str)

    print("행정동_with_코드.csv 읽는 중...")
    dong_map = pd.read_csv(MAP_PATH, dtype=str)

    hvac.columns = hvac.columns.str.strip()
    dong_map.columns = dong_map.columns.str.strip()

    print("\n[hvac 컬럼]")
    print(hvac.columns.tolist())

    print("\n[행정동 매핑 컬럼]")
    print(dong_map.columns.tolist())

    # =====================================================
    # 1. hvac 컬럼명 표준화
    # =====================================================
    hvac_rename = {
        "자치구": "gu",
        "자치구명": "gu",
        "구": "gu",

        "행정동": "dong",
        "행정동이름": "dong",
        "행정동명": "dong",
        "행정동_표준명": "dong",
        "동": "dong",

        "연도": "year",
        "년도": "year",
        "월": "month",
    }

    hvac = hvac.rename(columns=hvac_rename)

    # =====================================================
    # 2. 측정시간이 있으면 year/month 생성
    # =====================================================
    time_candidates = ["측정시간", "센서 시간", "datetime", "date", "일시", "등록 일시"]

    time_col = None
    for col in time_candidates:
        if col in hvac.columns:
            time_col = col
            break

    if time_col is not None:
        hvac[time_col] = pd.to_datetime(hvac[time_col], errors="coerce")
        time_na_mask = hvac[time_col].isna()

        print(f"{time_col} 변환 실패 rows:", int(time_na_mask.sum()))

        hvac = hvac.loc[~time_na_mask].copy()

        hvac["year"] = hvac[time_col].dt.year.astype(int)
        hvac["month"] = hvac[time_col].dt.month.astype(int)

    required_time_cols = ["year", "month"]
    missing_time = [c for c in required_time_cols if c not in hvac.columns]

    if missing_time:
        raise ValueError(
            f"hvac 원본에 year/month를 만들 수 없음: {missing_time}. "
            f"현재 컬럼: {hvac.columns.tolist()}"
        )

    # =====================================================
    # 3. 행정동 이름 수동 보정
    # stress 때랑 같은 이슈 방어
    # =====================================================
    manual_fix = {
        ("관악구", "삼성"): ("관악구", "삼성동"),
        ("강남구", "일원2동"): ("강남구", "일원본동"),
    }

    for (old_gu, old_dong), (new_gu, new_dong) in manual_fix.items():
        mask = (
            (hvac["gu"].astype(str).str.strip() == old_gu) &
            (hvac["dong"].astype(str).str.strip() == old_dong)
        )

        print(
            f"수동 보정 {old_gu} {old_dong} -> {new_gu} {new_dong}: "
            f"{int(mask.sum())} rows"
        )

        hvac.loc[mask, "gu"] = new_gu
        hvac.loc[mask, "dong"] = new_dong

    # =====================================================
    # 4. drop 대상
    # =====================================================
    drop_pairs = [
        ("동대문구", "전농3동"),
    ]

    drop_mask = pd.Series(False, index=hvac.index)

    for gu, dong in drop_pairs:
        pair_mask = (
            (hvac["gu"].astype(str).str.strip() == gu) &
            (hvac["dong"].astype(str).str.strip() == dong)
        )

        print(f"drop 대상 {gu} {dong}: {int(pair_mask.sum())} rows")
        drop_mask = drop_mask | pair_mask

    nan_mask = (
        hvac["gu"].isna() |
        hvac["dong"].isna() |
        (hvac["gu"].astype(str).str.strip().isin(["", "nan", "None"])) |
        (hvac["dong"].astype(str).str.strip().isin(["", "nan", "None"]))
    )

    print("자치구/행정동 NaN drop rows:", int(nan_mask.sum()))

    hvac = hvac.loc[~(drop_mask | nan_mask)].copy()

    # =====================================================
    # 5. 상일동 복제
    # =====================================================
    sangil_mask = (
        (hvac["gu"].astype(str).str.strip() == "강동구") &
        (hvac["dong"].astype(str).str.strip() == "상일동")
    )

    sangil_rows = hvac.loc[sangil_mask].copy()
    print("상일동 원본 rows:", len(sangil_rows))

    if len(sangil_rows) > 0:
        sangil_1 = sangil_rows.copy()
        sangil_2 = sangil_rows.copy()

        sangil_1["dong"] = "상일1동"
        sangil_2["dong"] = "상일2동"

        hvac = hvac.loc[~sangil_mask].copy()
        hvac = pd.concat([hvac, sangil_1, sangil_2], ignore_index=True)

        print("상일동 복제 후 추가 rows:", len(sangil_1) + len(sangil_2))
    

    # =====================================================
    # 추가. 구/동 불일치 보정
    # 기준: 행정동명을 신뢰하고, 실제 소속 자치구로 수정
    # 위치: 상일동 복제 후, 행정동코드 매칭 key 생성 전
    # =====================================================

    extra_gu_dong_fix = {
        ("강동구", "등촌1동"): ("강서구", "등촌1동"),
        ("관악구", "숭인1동"): ("종로구", "숭인1동"),
        ("관악구", "염창동"): ("강서구", "염창동"),
        ("양천구", "당산2동"): ("영등포구", "당산2동"),
        ("종로구", "동선동"): ("성북구", "동선동"),
        ("종로구", "신정1동"): ("양천구", "신정1동"),
        ("중구", "종로1·2·3·4가동"): ("종로구", "종로1·2·3·4가동"),
        ("중구", "창신1동"): ("종로구", "창신1동"),
    }

    for (bad_gu, bad_dong), (good_gu, good_dong) in extra_gu_dong_fix.items():
        mask = (
            (hvac["gu"].astype(str).str.strip() == bad_gu) &
            (hvac["dong"].astype(str).str.strip() == bad_dong)
        )

        print(
            f"추가 보정 {bad_gu} {bad_dong} -> {good_gu} {good_dong}: "
            f"{int(mask.sum())} rows"
        )

        hvac.loc[mask, "gu"] = good_gu
        hvac.loc[mask, "dong"] = good_dong


    # =====================================================
    # 추가. 남은 전농3동 drop
    # 위치: 추가 구/동 보정 후, 행정동코드 매칭 key 생성 전
    # =====================================================

    extra_drop_pairs = [
        ("종로구", "전농3동"),
        ("중구", "전농3동"),
    ]

    extra_drop_mask = pd.Series(False, index=hvac.index)

    for bad_gu, bad_dong in extra_drop_pairs:
        pair_mask = (
            (hvac["gu"].astype(str).str.strip() == bad_gu) &
            (hvac["dong"].astype(str).str.strip() == bad_dong)
        )

        print(f"추가 drop 대상 {bad_gu} {bad_dong}: {int(pair_mask.sum())} rows")

        extra_drop_mask = extra_drop_mask | pair_mask

    hvac = hvac.loc[~extra_drop_mask].copy()


    # =====================================================
    # 6. 매핑 파일 표준화
    # =====================================================
    map_rename = {
        "행정동코드": "code",
        "행정동_표준명": "dong",
        "자치구": "gu",
    }

    dong_map = dong_map.rename(columns=map_rename)

    hvac_required = ["gu", "dong"]
    map_required = ["gu", "dong", "code"]

    hvac_missing = [c for c in hvac_required if c not in hvac.columns]
    map_missing = [c for c in map_required if c not in dong_map.columns]

    if hvac_missing:
        raise ValueError(
            f"hvac 원본 필수 컬럼 없음: {hvac_missing}. "
            f"현재 컬럼: {hvac.columns.tolist()}"
        )

    if map_missing:
        raise ValueError(
            f"행정동_with_코드.csv 필수 컬럼 없음: {map_missing}. "
            f"현재 컬럼: {dong_map.columns.tolist()}"
        )

    hvac["gu_key"] = normalize_text(hvac["gu"])
    hvac["dong_key"] = normalize_text(hvac["dong"])

    dong_map["gu_key"] = normalize_text(dong_map["gu"])
    dong_map["dong_key"] = normalize_text(dong_map["dong"])
    dong_map["code"] = dong_map["code"].astype(str).str.strip()

    map_small = dong_map[
        ["gu_key", "dong_key", "code"]
    ].drop_duplicates()

    map_small = map_small.drop_duplicates(
        subset=["gu_key", "dong_key"],
        keep="first"
    )

    # 기존 code가 있으면 제거 후 다시 붙임
    if "code" in hvac.columns:
        print("기존 code 컬럼 제거 후 매핑 파일 기준으로 다시 생성")
        hvac = hvac.drop(columns=["code"])

    print("\n행정동코드 매칭 중...")

    merged = hvac.merge(
        map_small,
        on=["gu_key", "dong_key"],
        how="left"
    )

    fail = (
        merged[merged["code"].isna()][["gu", "dong"]]
        .drop_duplicates()
        .sort_values(["gu", "dong"])
    )

    print("\n전체 rows:", len(merged))
    print("매칭 실패 rows:", int(merged["code"].isna().sum()))
    print("매칭 실패 고유 조합 수:", len(fail))

    fail_path = BASE_DIR / "app/data/hvac_mapping_failed.csv"
    
    if len(fail) > 0:
        fail.to_csv(fail_path, index=False, encoding="utf-8-sig")
        print("매칭 실패 목록 저장:", fail_path)
    else:
        print("매칭 실패 없음")

    matched_codes = merged["code"].dropna().astype(str)

    print("\n행정동코드 길이 분포:")
    print(matched_codes.str.len().value_counts().sort_index())

    invalid_code = merged[
        merged["code"].notna()
        & (merged["code"].astype(str).str.len() != 10)
    ]

    if len(invalid_code) > 0:
        print("\n[경고] 10자리가 아닌 행정동코드 존재")
        print(
            invalid_code[["gu", "dong", "code"]]
            .drop_duplicates()
            .head(50)
        )

    merged = merged.drop(columns=["gu_key", "dong_key"])

    front_cols = ["code", "gu", "dong", "year", "month"]
    front_cols = [c for c in front_cols if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in front_cols]
    merged = merged[front_cols + other_cols]

    print("\n[저장 직전 컬럼]")
    print(merged.columns.tolist())

    required_output = ["code", "gu", "dong", "year", "month"]
    missing_output = [c for c in required_output if c not in merged.columns]

    if missing_output:
        raise ValueError(f"저장 직전 필수 컬럼 없음: {missing_output}")

    print("\n저장 중...")
    merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n완료:", OUTPUT_PATH)


if __name__ == "__main__":
    main()