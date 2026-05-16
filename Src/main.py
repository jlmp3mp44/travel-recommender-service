"""
FastAPI microservice for collaborative-filtering recommendations (SVD).
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from auto_retrain import PeriodicRetrainer
from svd_model import recommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

retrainer = PeriodicRetrainer(recommender)


@asynccontextmanager
async def lifespan(_: FastAPI):
    retrainer.start()
    try:
        yield
    finally:
        retrainer.stop()


app = FastAPI(title="Recommender Service", version="1.0.0", lifespan=lifespan)


# ── Request / Response schemas ───────────────────────────────────────

class PredictRequest(BaseModel):
    user_id: int
    place_ids: List[int]


class PredictResponse(BaseModel):
    predictions: Dict[int, float]


class RatingItem(BaseModel):
    user_id: int
    place_id: int
    rating: float = Field(ge=1, le=5)


class RetrainRequest(BaseModel):
    ratings: List[RatingItem]


class RetrainResponse(BaseModel):
    status: str
    n_ratings: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=recommender.is_trained())


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    logger.info("Predict request for user %s on %d places", req.user_id, len(req.place_ids))
    predictions = recommender.predict(req.user_id, req.place_ids)
    return PredictResponse(predictions=predictions)


@app.post("/retrain", response_model=RetrainResponse)
def retrain(req: RetrainRequest) -> RetrainResponse:
    logger.info("Retrain request with %d ratings", len(req.ratings))
    try:
        ratings_data = [r.model_dump() for r in req.ratings]
        recommender.train(ratings_data)
        return RetrainResponse(status="ok", n_ratings=len(req.ratings))
    except Exception as exc:
        logger.exception("Retrain failed")
        raise HTTPException(status_code=500, detail=str(exc))
