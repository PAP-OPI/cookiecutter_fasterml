import os
import pickle
import sys
import time

sys.path.append(os.path.abspath(os.curdir))

import pandas as pd
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from base_class_data import Config
from main import {{cookiecutter.model_name}}


app: FastAPI = FastAPI()

REQUEST_COUNT = Counter(
    "request_count", "App Request Count", ["method", "endpoint", "http_status"]
)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Request latency in seconds", ["endpoint"]
)


@app.middleware("http")
async def add_prometheus_metrics(request: Request, call_next):
    """
    Middleware to collect Prometheus metrics for request counts and latency.
    """
    start_time: float = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    global REQUEST_COUNT
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        http_status=response.status_code,
    ).inc()

    global REQUEST_LATENCY
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(process_time)

    return response


@app.get("/health")
async def health_check():
    """
    Health check endpoint to ensure the API is running.
    """
    return {"status": "Todo Gucci"}


@app.get("/metrics")
async def metrics():
    """
    Expose Prometheus metrics at the /metrics endpoint.
    """
    return Response(generate_latest(), media_type="text/plain")


@app.post("/predict")
async def predict(data: Config):
    """
    Dummy predict endpoint.
    Replace with your actual model prediction logic.
    """
    pre = {{cookiecutter.model_name}}()
    df = pd.DataFrame.from_records([data.__dict__])
    df = pre.preprocess_data(df)
    
    with open(os.path.join(os.curdir, '.artifacts/model.pkl'), 'rb') as file:
        model = pickle.load(file)

    cols = model.feature_names_in_

    df = df[cols]

    prediction = model.predict(df)

    return {"result": str(prediction[0])}
