from pathlib import Path

from app.services import hvac_service_model


BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "app/data/api"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    year = 2026
    month = 5

    print(f"hvac 예측 결과 생성 중: {year}-{month:02d}")

    df = hvac_service_model.get_heatmap(year, month)

    print("shape:", df.shape)
    print(df.head())

    out_path = OUT_DIR / f"hvac_{year}_{month:02d}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("저장 완료:", out_path)


if __name__ == "__main__":
    main()