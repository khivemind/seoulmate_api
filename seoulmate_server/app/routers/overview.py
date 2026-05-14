from fastapi import APIRouter, HTTPException, Header, Query

from app.services.registry import LAYER_SERVICES


router = APIRouter()

VALID_API_KEY = "v9WzP1xF7K8lQ2mR4sT6uY8aB0cD3eF9GhJkLmNo"


def normalize_code(code) -> str:
    return str(code).replace(".0", "").strip()


@router.get("/v1/overview/{code}")
def get_overview(
    code: str,
    year: int = Query(2026),
    month: int = Query(5),
    x_api_key: str | None = Header(default=None),
):
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="인증 실패")

    code = normalize_code(code)

    required_layers = ["safety", "comfort", "hvac", "expenses", "health", "stress"]

    layer_scores = {
        "safety": 0,
        "comfort": 0,
        "hvac": 0,
        "expenses": 0,
        "health": 0,
        "stress": 0,
    }

    dong = ""
    gu = ""

    for layer in required_layers:
        service = LAYER_SERVICES.get(layer)

        if service is None:
            print(f"[overview] service 없음: {layer}")
            continue

        try:
            df = service.get_heatmap(year, month)
        except Exception as e:
            print(f"[overview] {layer} service error:", repr(e))
            continue

        if df.empty:
            print(f"[overview] {layer} empty: year={year}, month={month}")
            continue

        if "code" not in df.columns:
            print(f"[overview] {layer} code 컬럼 없음")
            continue

        df = df.copy()
        df["code"] = df["code"].apply(normalize_code)

        matched = df[df["code"] == code]

        if matched.empty:
            print(f"[overview] {layer} code 매칭 실패: {code}")
            continue

        row = matched.iloc[0]

        layer_scores[layer] = float(row.get("score", 0))

        if not dong:
            dong = str(row.get("dong", ""))

        if not gu:
            gu = str(row.get("gu", ""))

    valid_scores = [
        score
        for score in layer_scores.values()
        if score not in [None, 0]
    ]

    if valid_scores:
        average_score = round(sum(valid_scores) / len(valid_scores))
    else:
        average_score = 0

    return {
        "status": 200,
        "code": code,
        "dong": dong,
        "gu": gu,
        "score": average_score,
        "safety": round(layer_scores.get("safety", 0)),
        "comfort": round(layer_scores.get("comfort", 0)),
        "hvac": round(layer_scores.get("hvac", 0)),
        "expenses": round(layer_scores.get("expenses", 0)),
        "health": round(layer_scores.get("health", 0)),
        "stress": round(layer_scores.get("stress", 0)),
        "average_score": average_score,
        "score_last_year": [],
        "recent_trend": 0,
    }


def calculate_trend(scores: list[int]) -> int:
    """
    0 유지
    1 상승
    2 하강
    """
    if len(scores) < 2:
        return 0

    before = scores[-2]
    current = scores[-1]

    if current > before:
        return 1
    elif current < before:
        return 2
    else:
        return 0