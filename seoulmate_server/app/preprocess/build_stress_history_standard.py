from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_PATH = BASE_DIR / "app/data/stress_history.csv"
MAP_PATH = BASE_DIR / "app/data/행정동_with_코드.csv"
OUTPUT_PATH = BASE_DIR / "app/data/stress_history_standard.csv"


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
    print("원본 stress_history.csv 읽는 중...")
    stress = pd.read_csv(RAW_PATH, dtype=str)

    print("행정동_with_코드.csv 읽는 중...")
    dong_map = pd.read_csv(MAP_PATH, dtype=str)

    stress.columns = stress.columns.str.strip()
    dong_map.columns = dong_map.columns.str.strip()

    print("\n[stress 컬럼]")
    print(stress.columns.tolist())

    print("\n[행정동 매핑 컬럼]")
    print(dong_map.columns.tolist())

    # =====================================================
    # 1. stress_history.csv 컬럼명 표준화
    # =====================================================
    stress_rename = {
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

    stress = stress.rename(columns=stress_rename)

    # =====================================================
    # 추가. 측정시간 기준 year/month 생성
    # 위치: stress 컬럼명 표준화 직후
    # =====================================================

    if "측정시간" not in stress.columns:
        raise ValueError(
            f"stress_history.csv에 '측정시간' 컬럼이 없음. 현재 컬럼: {stress.columns.tolist()}"
        )

    stress["측정시간"] = pd.to_datetime(stress["측정시간"], errors="coerce")

    time_na_mask = stress["측정시간"].isna()
    print("측정시간 변환 실패 rows:", int(time_na_mask.sum()))

    # 측정시간이 없는 행은 year/month를 만들 수 없으므로 제거
    stress = stress.loc[~time_na_mask].copy()

    stress["year"] = stress["측정시간"].dt.year.astype(int)
    stress["month"] = stress["측정시간"].dt.month.astype(int)

    print("year 분포:")
    print(stress["year"].value_counts().sort_index())

    print("month 분포:")
    print(stress["month"].value_counts().sort_index())

    # =====================================================
    # 추가. 행정동 이름 수동 보정
    # 위치: stress 컬럼명 표준화 직후, key 생성 전
    # =====================================================

    manual_fix = {
        # 오기 수정
        ("관악구", "삼성"): ("관악구", "삼성동"),

        # 네 매핑 파일 기준:
        # 원본_행정동_한글 = 일원2동
        # 행정동_표준명 = 일원본동
        # 행정동코드 = 1168072000
        ("강남구", "일원2동"): ("강남구", "일원본동"),
    }

    for (old_gu, old_dong), (new_gu, new_dong) in manual_fix.items():
        mask = (
            (stress["gu"].astype(str).str.strip() == old_gu) &
            (stress["dong"].astype(str).str.strip() == old_dong)
        )

        print(
            f"수동 보정 {old_gu} {old_dong} -> {new_gu} {new_dong}: "
            f"{int(mask.sum())} rows"
        )

        stress.loc[mask, "gu"] = new_gu
        stress.loc[mask, "dong"] = new_dong


    # =====================================================
    # 추가. 매칭 불가/폐지 행정동 drop
    # =====================================================

    drop_pairs = [
        ("동대문구", "전농3동"),
    ]

    drop_mask = pd.Series(False, index=stress.index)

    for gu, dong in drop_pairs:
        pair_mask = (
            (stress["gu"].astype(str).str.strip() == gu) &
            (stress["dong"].astype(str).str.strip() == dong)
        )

        print(f"drop 대상 {gu} {dong}: {int(pair_mask.sum())} rows")

        drop_mask = drop_mask | pair_mask


    # NaN / 빈 문자열 제거
    nan_mask = (
        stress["gu"].isna() |
        stress["dong"].isna() |
        (stress["gu"].astype(str).str.strip().isin(["", "nan", "None"])) |
        (stress["dong"].astype(str).str.strip().isin(["", "nan", "None"]))
    )

    print("자치구/행정동 NaN drop rows:", int(nan_mask.sum()))

    stress = stress.loc[~(drop_mask | nan_mask)].copy()

    # =====================================================
    # 추가. 강동구 상일동 데이터 복제
    # 상일동은 상일1동/상일2동에 동일 적용
    # =====================================================

    sangil_mask = (
        (stress["gu"].astype(str).str.strip() == "강동구") &
        (stress["dong"].astype(str).str.strip() == "상일동")
    )

    sangil_rows = stress.loc[sangil_mask].copy()

    print("상일동 원본 rows:", len(sangil_rows))

    if len(sangil_rows) > 0:
        sangil_1 = sangil_rows.copy()
        sangil_2 = sangil_rows.copy()

        sangil_1["dong"] = "상일1동"
        sangil_2["dong"] = "상일2동"

        # 원본 상일동 제거 후 상일1동/상일2동으로 복제
        stress = stress.loc[~sangil_mask].copy()
        stress = pd.concat([stress, sangil_1, sangil_2], ignore_index=True)

        print("상일동 복제 후 추가 rows:", len(sangil_1) + len(sangil_2))


    # ============================================================
    # 추가 구/동 불일치 보정
    # 기준: 행정동명을 신뢰하고, 실제 소속 자치구로 수정
    # 위치: 행정동코드 merge 직전
    # ============================================================

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
        mask = (stress["gu"] == bad_gu) & (stress["dong"] == bad_dong)
        print(
            f"추가 보정 {bad_gu} {bad_dong} -> {good_gu} {good_dong}: "
            f"{mask.sum()} rows"
        )
        stress.loc[mask, "gu"] = good_gu
        stress.loc[mask, "dong"] = good_dong


    # ============================================================
    # 존재하지 않거나 현재 행정동코드 매핑 불가한 동 drop
    # ============================================================

    extra_drop_pairs = [
        ("종로구", "전농3동"),
        ("중구", "전농3동"),
    ]

    for bad_gu, bad_dong in extra_drop_pairs:
        mask = (stress["gu"] == bad_gu) & (stress["dong"] == bad_dong)
        print(f"추가 drop 대상 {bad_gu} {bad_dong}: {mask.sum()} rows")
        stress = stress.loc[~mask].copy()


    # =====================================================
    # 2. 행정동_with_코드.csv 컬럼명 표준화
    # =====================================================
    map_rename = {
        "행정동코드": "code",
        "행정동_표준명": "dong",
        "자치구": "gu",
    }

    dong_map = dong_map.rename(columns=map_rename)

    # =====================================================
    # 3. 필수 컬럼 검사
    # =====================================================
    stress_required = ["gu", "dong"]
    map_required = ["gu", "dong", "code"]

    stress_missing = [c for c in stress_required if c not in stress.columns]
    map_missing = [c for c in map_required if c not in dong_map.columns]

    if stress_missing:
        raise ValueError(
            f"stress_history.csv에 필요한 컬럼이 없음: {stress_missing}. "
            f"현재 컬럼: {stress.columns.tolist()}"
        )

    if map_missing:
        raise ValueError(
            f"행정동_with_코드.csv에 필요한 컬럼이 없음: {map_missing}. "
            f"현재 컬럼: {dong_map.columns.tolist()}"
        )

    # =====================================================
    # 4. 매칭용 key 생성
    # =====================================================
    stress["gu_key"] = normalize_text(stress["gu"])
    stress["dong_key"] = normalize_text(stress["dong"])

    dong_map["gu_key"] = normalize_text(dong_map["gu"])
    dong_map["dong_key"] = normalize_text(dong_map["dong"])
    dong_map["code"] = dong_map["code"].astype(str).str.strip()

    # =====================================================
    # 5. 매핑 테이블 정리
    # =====================================================
    map_small = dong_map[
        ["gu_key", "dong_key", "code"]
    ].drop_duplicates()

    duplicated = map_small.duplicated(
        subset=["gu_key", "dong_key"],
        keep=False
    )

    if duplicated.any():
        print("\n[경고] 매핑 파일에 중복된 자치구+행정동 조합이 있음")
        print(
            map_small[duplicated]
            .sort_values(["gu_key", "dong_key"])
            .head(50)
        )

    map_small = map_small.drop_duplicates(
        subset=["gu_key", "dong_key"],
        keep="first"
    )

    # =====================================================
    # 6. merge
    # =====================================================
    print("\n행정동코드 매칭 중...")

    merged = stress.merge(
        map_small,
        on=["gu_key", "dong_key"],
        how="left"
    )

    # =====================================================
    # 7. 매칭 실패 확인
    # =====================================================
    fail = (
        merged[merged["code"].isna()][["gu", "dong"]]
        .drop_duplicates()
        .sort_values(["gu", "dong"])
    )

    print("\n전체 rows:", len(merged))
    print("매칭 실패 rows:", merged["code"].isna().sum())
    print("매칭 실패 고유 조합 수:", len(fail))

    if len(fail) > 0:
        print("\n매칭 실패 예시:")
        print(fail.head(50))

        fail_path = BASE_DIR / "app/data/stress_mapping_failed.csv"
        fail.to_csv(fail_path, index=False, encoding="utf-8-sig")

        print("\n매칭 실패 목록 저장:", fail_path)

    # =====================================================
    # 8. code 10자리 검증
    # =====================================================
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

    # =====================================================
    # 9. 정리 후 저장
    # =====================================================
    merged = merged.drop(columns=["gu_key", "dong_key"])

    front_cols = ["code", "gu", "dong"]
    other_cols = [c for c in merged.columns if c not in front_cols]
    merged = merged[front_cols + other_cols]

    print("\n저장 중...")
    merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n완료:", OUTPUT_PATH)


if __name__ == "__main__":
    main()