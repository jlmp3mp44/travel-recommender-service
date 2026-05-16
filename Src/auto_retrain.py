"""
Periodic SVD retraining from PostgreSQL activity ratings.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, List

import psycopg2
from psycopg2.extensions import connection as PgConnection

from svd_model import SVDRecommender

logger = logging.getLogger(__name__)


SQL_RATINGS_EXPORT = """
SELECT
  ar.user_id AS user_id,
  ap.place_id AS place_id,
  MAX(ar.stars)::float8 AS rating
FROM activity_ratings ar
JOIN activity_places ap ON ap.activity_id = ar.activity_id
GROUP BY ar.user_id, ap.place_id
ORDER BY ar.user_id, ap.place_id
"""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%r, using default %d", name, raw, default)
        return default
    return max(minimum, value)


class PeriodicRetrainer:
    """
    Periodically exports ratings from DB and retrains the in-memory recommender.
    """

    def __init__(self, recommender: SVDRecommender) -> None:
        self._recommender = recommender
        self._enabled = _env_bool("SVD_AUTO_RETRAIN_ENABLED", True)
        self._initial_delay_seconds = _env_int("SVD_AUTO_RETRAIN_INITIAL_DELAY_SECONDS", 60)
        self._interval_seconds = _env_int("SVD_AUTO_RETRAIN_INTERVAL_SECONDS", 3600, minimum=10)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self._enabled:
            logger.info("Periodic SVD retraining is disabled (SVD_AUTO_RETRAIN_ENABLED=false)")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="svd-auto-retrainer",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Periodic SVD retrainer started: initial_delay=%ss interval=%ss",
            self._initial_delay_seconds,
            self._interval_seconds,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _run_loop(self) -> None:
        if self._initial_delay_seconds > 0 and self._stop_event.wait(self._initial_delay_seconds):
            return
        while not self._stop_event.is_set():
            try:
                self._run_once()
            except Exception:
                logger.exception("Scheduled SVD retrain failed")
            if self._stop_event.wait(self._interval_seconds):
                return

    def _run_once(self) -> None:
        ratings = self._load_ratings_from_db()
        if not ratings:
            logger.info("No ratings found for scheduled SVD retraining")
            return
        logger.info("Scheduled SVD retrain with %d user-place ratings", len(ratings))
        self._recommender.train(ratings)

    def _load_ratings_from_db(self) -> List[Dict[str, float]]:
        with self._open_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(SQL_RATINGS_EXPORT)
                rows = cur.fetchall()
        return [
            {
                "user_id": int(user_id),
                "place_id": int(place_id),
                "rating": float(rating),
            }
            for (user_id, place_id, rating) in rows
        ]

    def _open_connection(self) -> PgConnection:
        dsn = os.getenv("RECOMMENDER_DB_DSN")
        if dsn:
            return psycopg2.connect(dsn)

        host = os.getenv("RECOMMENDER_DB_HOST", "localhost")
        port = _env_int("RECOMMENDER_DB_PORT", 5432, minimum=1)
        dbname = os.getenv("RECOMMENDER_DB_NAME", "explorer")
        user = os.getenv("RECOMMENDER_DB_USER", "postgres")
        password = os.getenv("RECOMMENDER_DB_PASSWORD", "")
        sslmode = os.getenv("RECOMMENDER_DB_SSLMODE", "prefer")
        return psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            sslmode=sslmode,
        )
