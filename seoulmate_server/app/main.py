from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import heatmap, overview

app = FastAPI(
    title="SeoulMate Smart City AI Server",
    description="스마트시티 히트맵 기반 통합 AI 서버",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포할 때는 프론트 주소로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(heatmap.router)
app.include_router(overview.router)


@app.get("/")
def root():
    return {
        "service": "SeoulMate Smart City AI Server",
        "status": "running",
        "layers": [
            "overall",
            "safety",
            "health",
            "stress",
            "hvac",
            "comfort",
            "expenses"
        ]
    }