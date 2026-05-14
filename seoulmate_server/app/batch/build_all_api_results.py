from pathlib import Path

from app.services import stress_service_model
from app.services import hvac_service_model
from app.services import comfort_service_rule
from app.services import health_service_rule
from app.services import safety_service_rule


BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "app/data/api"
OUT_DIR.mkdir(parents=True, exist_ok=True)


SERVICES = {
    "stress": stress_service_model,
    "hvac": hvac_service_model,
    "comfort": comfort_service_rule,
    "health": health_service_rule,
    "safety": safety_service_rule,
}


def save_layer(layer: str, service, year: int, month: int):
    print(f"\n[{layer}] 생성 중...")

    df = service.get_heatmap(year, month)

    print("shape:", df.shape)

    out_path = OUT_DIR / f"{layer}_{year}_{month:02d}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("저장 완료:", out_path)


def main():
    year = 2026
    month = 5

    for layer, service in SERVICES.items():
        save_layer(layer, service, year, month)


if __name__ == "__main__":
    main()