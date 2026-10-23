import logging
import sys
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "artifacts" / "animal_pipeline.joblib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("animal-classification-api")

FEATURE_FIELDS = [
    "hair", "feathers", "eggs", "milk", "airborne", "aquatic",
    "predator", "toothed", "backbone", "breathes", "venomous",
    "fins", "legs", "tail", "domestic", "catsize",
]

class AnimalFeatures(BaseModel):
    hair: int = Field(ge=0, le=1)
    feathers: int = Field(ge=0, le=1)
    eggs: int = Field(ge=0, le=1)
    milk: int = Field(ge=0, le=1)
    airborne: int = Field(ge=0, le=1)
    aquatic: int = Field(ge=0, le=1)
    predator: int = Field(ge=0, le=1)
    toothed: int = Field(ge=0, le=1)
    backbone: int = Field(ge=0, le=1)
    breathes: int = Field(ge=0, le=1)
    venomous: int = Field(ge=0, le=1)
    fins: int = Field(ge=0, le=1)
    legs: int = Field(ge=0, le=8)
    tail: int = Field(ge=0, le=1)
    domestic: int = Field(ge=0, le=1)
    catsize: int = Field(ge=0, le=1)

app = FastAPI(
    title="Animal Classification API",
    description="Predicts the animal class using the trained Zoo dataset pipeline.",
    version="1.0.0",
)

try:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)
    logger.info("event=model_loaded model_file=%s", MODEL_FILE.name)
except Exception:
    logger.exception("event=model_load_failed model_file=%s", MODEL_FILE.name)
    raise

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        level = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(
            level,
            "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id, request.method, request.url.path,
            response.status_code, duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "event=http_request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id, request.method, request.url.path, duration_ms,
        )
        raise

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    fields = [
        ".".join(str(part) for part in error.get("loc", []))
        for error in exc.errors()
    ]
    logger.warning(
        "event=validation_error path=%s field_locations=%s error_count=%s",
        request.url.path, fields, len(exc.errors()),
    )
    return await request_validation_exception_handler(request, exc)

@app.get("/")
def root():
    return {"message": "Animal Classification API is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: AnimalFeatures):
    try:
        payload = data.model_dump()
        input_df = pd.DataFrame(
            [[payload[field] for field in FEATURE_FIELDS]],
            columns=FEATURE_FIELDS,
        )
        logger.info("event=prediction_request feature_names=%s", FEATURE_FIELDS)
        prediction = model.predict(input_df)[0]
        logger.info("event=prediction_success prediction_class=%s", str(prediction))
        return {"predicted_class_type": int(prediction)}
    except Exception as exc:
        logger.exception("event=prediction_failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
