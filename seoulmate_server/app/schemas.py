from pydantic import BaseModel
from typing import List


class DongScore(BaseModel):
    code: str
    dong: str
    gu: str
    grade: int
    score: float


class HeatmapResponse(BaseModel):
    status: int
    dong_list: List[DongScore]


class OverviewResponse(BaseModel):
    status: int
    code: str
    dong: str
    gu: str
    score: int
    safety: int
    comfort: int
    hvac: int
    expenses: int
    health: int
    stress: int
    average_score: int
    score_last_year: List[int]
    recent_trend: int