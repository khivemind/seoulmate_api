from pathlib import Path

import pandas as pd


GU_ENG_TO_KOR = {
    "Gangnam-gu": "강남구",
    "Gangdong-gu": "강동구",
    "Gangbuk-gu": "강북구",
    "Gangseo-gu": "강서구",
    "Gwanak-gu": "관악구",
    "Gwangjin-gu": "광진구",
    "Guro-gu": "구로구",
    "Geumcheon-gu": "금천구",
    "Nowon-gu": "노원구",
    "Dobong-gu": "도봉구",
    "Dongdaemun-gu": "동대문구",
    "Dongjak-gu": "동작구",
    "Mapo-gu": "마포구",
    "Seodaemun-gu": "서대문구",
    "Seocho-gu": "서초구",
    "Seongdong-gu": "성동구",
    "Seongbuk-gu": "성북구",
    "Songpa-gu": "송파구",
    "Yangcheon-gu": "양천구",
    "Yeongdeungpo-gu": "영등포구",
    "Yongsan-gu": "용산구",
    "Eunpyeong-gu": "은평구",
    "Jongno-gu": "종로구",
    "Jung-gu": "중구",
    "Jungnang-gu": "중랑구",
    "Seoul_Grand_Park": "서울대공원",
}


def read_csv_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="cp949")


def normalize_regions_with_dong_csv(
    raw_df: pd.DataFrame,
    dong_map_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    새로 들어온 센서 데이터의 영어 자치구/행정동을 한글 자치구/행정동으로 변환한다.

    raw_df 필요 컬럼:
    - 자치구
    - 행정동

    dong_map_df 필요 컬럼:
    - 원본_자치구 또는 자치구
    - 행정동_영문_원본
    - 원본_행정동_한글
    """
    df = raw_df.copy()
    dong_map = dong_map_df.copy()

    df.columns = df.columns.str.strip()
    dong_map.columns = dong_map.columns.str.strip()

    required_raw_cols = ["자치구", "행정동"]
    missing_raw = [c for c in required_raw_cols if c not in df.columns]
    if missing_raw:
        raise ValueError(f"새 센서 데이터에 필수 컬럼 없음: {missing_raw}, 현재 컬럼: {df.columns.tolist()}")

    if "원본_자치구" not in dong_map.columns:
        if "자치구" in dong_map.columns:
            dong_map["원본_자치구"] = dong_map["자치구"]
        else:
            raise ValueError(
                "행정동.csv에 원본_자치구 또는 자치구 컬럼이 필요함. "
                f"현재 컬럼: {dong_map.columns.tolist()}"
            )

    required_map_cols = ["원본_자치구", "행정동_영문_원본", "원본_행정동_한글"]
    missing_map = [c for c in required_map_cols if c not in dong_map.columns]
    if missing_map:
        raise ValueError(f"행정동.csv 필수 컬럼 없음: {missing_map}, 현재 컬럼: {dong_map.columns.tolist()}")

    df["자치구"] = df["자치구"].astype(str).str.strip()
    df["행정동"] = df["행정동"].astype(str).str.strip()

    # 원본 추적용
    df["자치구_영문_원본"] = df["자치구"]
    df["행정동_영문_원본"] = df["행정동"]

    # 1. 자치구 영어 → 한글
    df["자치구"] = df["자치구"].map(GU_ENG_TO_KOR).fillna(df["자치구"])

    # 2. 행정동 영어 → 한글
    mapping_df = (
        dong_map[["원본_자치구", "행정동_영문_원본", "원본_행정동_한글"]]
        .dropna(subset=["원본_자치구", "행정동_영문_원본", "원본_행정동_한글"])
        .drop_duplicates()
        .copy()
    )

    mapping_df["원본_자치구"] = mapping_df["원본_자치구"].astype(str).str.strip()
    mapping_df["행정동_영문_원본"] = mapping_df["행정동_영문_원본"].astype(str).str.strip()
    mapping_df["원본_행정동_한글"] = mapping_df["원본_행정동_한글"].astype(str).str.strip()

    mapping_series = mapping_df.set_index(
        ["원본_자치구", "행정동_영문_원본"]
    )["원본_행정동_한글"]

    keys = pd.MultiIndex.from_arrays(
        [df["자치구"], df["행정동_영문_원본"]],
        names=["원본_자치구", "행정동_영문_원본"],
    )

    mapped_dong = pd.Series(
        mapping_series.reindex(keys).to_numpy(),
        index=df.index,
    )

    df["행정동"] = mapped_dong.where(
        mapped_dong.notna() & (mapped_dong != ""),
        df["행정동"],
    )

    # 3. 서울대공원 제거
    before = len(df)
    df = df[df["자치구"] != "서울대공원"].copy()
    print(f"[normalize] 서울대공원 제거 rows: {before - len(df):,}")

    # 4. 매핑 실패 확인
    eng_gu_count = df["자치구"].astype(str).str.contains("-gu|Seoul_", regex=True, na=False).sum()
    eng_dong_count = df["행정동"].astype(str).str.contains("-dong|_", regex=True, na=False).sum()

    print(f"[normalize] 영문 자치구 잔여 rows: {eng_gu_count:,}")
    print(f"[normalize] 영문 행정동 잔여 rows: {eng_dong_count:,}")

    failed = df[
        df["행정동"].astype(str).str.contains("-dong|_", regex=True, na=False)
    ][["자치구_영문_원본", "행정동_영문_원본", "자치구", "행정동"]].drop_duplicates()

    return df, failed