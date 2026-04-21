"""
SVDRecommender — collaborative filtering model using surprise SVD.
"""

import logging
import threading
from typing import Dict, List

import pandas as pd
from surprise import SVD, Dataset, Reader

logger = logging.getLogger(__name__)


class SVDRecommender:
    """Wraps the surprise SVD algorithm for rating prediction."""

    def __init__(self) -> None:
        self._model: SVD | None = None
        self._trainset = None
        self._lock = threading.Lock()

    def is_trained(self) -> bool:
        return self._model is not None

    def train(self, ratings_data: List[dict]) -> None:
        """Build a surprise Dataset from *ratings_data* and train an SVD model.

        Each item in *ratings_data* must have keys: user_id, place_id, rating (1-5).
        """
        with self._lock:
            logger.info("Training SVD model with %d ratings…", len(ratings_data))
            df = pd.DataFrame(ratings_data)[["user_id", "place_id", "rating"]]
            reader = Reader(rating_scale=(1, 5))
            dataset = Dataset.load_from_df(df, reader)
            trainset = dataset.build_full_trainset()

            model = SVD()
            model.fit(trainset)

            self._model = model
            self._trainset = trainset
            logger.info("SVD model trained successfully.")

    def predict(self, user_id: int, place_ids: List[int]) -> Dict[int, float]:
        """Return predicted ratings for *user_id* on each of *place_ids*.

        Returns an empty dict when the model is not trained or the user is
        unknown (cold-start).
        """
        with self._lock:
            if not self.is_trained():
                logger.warning("predict() called but model is not trained.")
                return {}

            try:
                self._trainset.to_inner_uid(user_id)
            except ValueError:
                logger.warning("User %s unknown (cold start) — returning empty predictions.", user_id)
                return {}

            predictions: Dict[int, float] = {}
            for place_id in place_ids:
                pred = self._model.predict(user_id, place_id)
                predictions[place_id] = round(float(pred.est), 4)
            return predictions


# Module-level singleton
recommender = SVDRecommender()
