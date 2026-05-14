from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
API_DATA_DIR = BASE_DIR / "app/data/api"


def empty_result() -> pd.DataFrame:
    return pd.DataFrame(columns=["code", "dong", "gu", "grade", "score"])


def load_layer_result(layer: str, year: int, month: int) -> pd.DataFrame:
    path = API_DATA_DIR / f"{layer}_{int(year)}_{int(month):02d}.csv"

    if not path.exists():
        print(f"[{layer}] API 결과 파일 없음: {path}")
        return empty_result()

    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()

    required_cols = ["code", "dong", "gu", "grade", "score"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"{path.name} 필수 컬럼 누락: {missing}")

    df = df[required_cols].copy()

    df["code"] = df["code"].astype(str).str.replace(".0", "", regex=False).str.strip()
    df["dong"] = df["dong"].astype(str).str.strip()
    df["gu"] = df["gu"].astype(str).str.strip()
    df["grade"] = pd.to_numeric(df["grade"], errors="coerce").fillna(0).astype(int)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).round(2)

    return df