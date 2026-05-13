from fastapi import APIRouter, Query, HTTPException, Header
from app.schemas import HeatmapResponse, DongScore
from app.services.registry import get_service, LAYER_SERVICES

router = APIRouter()


VALID_API_KEY = "v9WzP1xF7K8lQ2mR4sT6uY8aB0cD3eF9GhJkLmNo"


@router.get("/v1/heatmap", response_model=HeatmapResponse)
def get_heatmap(
    layer: str = Query(...),
    year: int = Query(...),
    month: int = Query(...),
    x_api_key: str | None = Header(default=None)
):
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="인증 실패")

    if month < 1 or month > 12:
        raise HTTPException(status_code=400, detail="month는 1~12 사이여야 함")

    if layer == "overall":
        df = get_overall_heatmap(year, month)
    else:
        service = get_service(layer)

        if service is None:
            raise HTTPException(status_code=400, detail="잘못된 layer 값")

        df = service.get_heatmap(year, month)

    dong_list = [
        DongScore(
            code=str(row["code"]),
            dong=str(row["dong"]),
            gu=str(row["gu"]),
            grade=int(row["grade"]),
            score=float(row["score"]),
        )
        for _, row in df.iterrows()
    ]

    return HeatmapResponse(
        status=200,
        dong_list=dong_list
    )


def get_overall_heatmap(year: int, month: int):
    """
    전체 종합 점수.
    overall score는 '높을수록 살기 좋음 / 추천도 높음' 기준으로 계산한다.

    comfort, safety:
        높을수록 좋음 → 그대로 사용

    stress, hvac, health, expenses:
        높을수록 부담/위험 큼 → 100 - score로 변환해서 사용
    """
    import pandas as pd

    dfs = []

    reverse_layers = {"stress", "hvac", "health", "expenses"}

    for layer, service in LAYER_SERVICES.items():
        df = service.get_heatmap(year, month)

        if df.empty:
            continue

        temp = df[["code", "dong", "gu", "score"]].copy()

        if layer in reverse_layers:
            temp["score"] = 100 - temp["score"]

        temp = temp.rename(columns={"score": layer})
        dfs.append(temp)

    if not dfs:
        return pd.DataFrame(columns=["code", "dong", "gu", "grade", "score"])

    base = dfs[0]

    for df in dfs[1:]:
        base = base.merge(df, on=["code", "dong", "gu"], how="outer")

    score_cols = [
        col for col in base.columns
        if col not in ["code", "dong", "gu"]
    ]

    base["score"] = base[score_cols].mean(axis=1).round(2)

    def score_to_grade(score):
        if score <= 20:
            return 1
        elif score <= 40:
            return 2
        elif score <= 60:
            return 3
        elif score <= 80:
            return 4
        else:
            return 5

    base["grade"] = base["score"].apply(score_to_grade)

    return base[["code", "dong", "gu", "grade", "score"]]

    def score_to_grade(score):
        if score <= 20:
            return 1
        elif score <= 40:
            return 2
        elif score <= 60:
            return 3
        elif score <= 80:
            return 4
        else:
            return 5

    base["grade"] = base["score"].apply(score_to_grade)

    return base[["code", "dong", "gu", "grade", "score"]]