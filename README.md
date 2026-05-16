# travel-recommender-service

--------------

#### python 3.11 ####

```pip install -r requirements.txt```

```uvicorn main:app --host 0.0.0.0 --port 8001``` travel explorer expects port 8001

## Automatic DB-based retraining

This service can retrain itself on a schedule by pulling ratings directly from PostgreSQL
(`activity_ratings` + `activity_places`).

Environment variables:

- `SVD_AUTO_RETRAIN_ENABLED` (default: `true`)
- `SVD_AUTO_RETRAIN_INITIAL_DELAY_SECONDS` (default: `60`)
- `SVD_AUTO_RETRAIN_INTERVAL_SECONDS` (default: `3600`)
- `RECOMMENDER_DB_DSN` (optional full DSN, takes precedence)
- `RECOMMENDER_DB_HOST` (default: `localhost`)
- `RECOMMENDER_DB_PORT` (default: `5432`)
- `RECOMMENDER_DB_NAME` (default: `explorer`)
- `RECOMMENDER_DB_USER` (default: `postgres`)
- `RECOMMENDER_DB_PASSWORD` (default: empty)
- `RECOMMENDER_DB_SSLMODE` (default: `prefer`)

-----------
#cmd#

pip install -r requirements.txt

set RECOMMENDER_DB_HOST=localhost
set RECOMMENDER_DB_PORT=5432
set RECOMMENDER_DB_NAME=explorer
set RECOMMENDER_DB_USER=postgres
set RECOMMENDER_DB_PASSWORD=YOUR_DB_PASSWORD
set RECOMMENDER_DB_SSLMODE=prefer

set SVD_AUTO_RETRAIN_ENABLED=true
set SVD_AUTO_RETRAIN_INITIAL_DELAY_SECONDS=60
set SVD_AUTO_RETRAIN_INTERVAL_SECONDS=3600

uvicorn main:app --host 0.0.0.0 --port 8001
