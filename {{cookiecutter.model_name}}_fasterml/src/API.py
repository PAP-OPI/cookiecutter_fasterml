import os
import pickle
import sys
import time
import sqlite3

#Memo test
import psutil

sys.path.append(os.path.abspath(os.curdir))

import pandas as pd
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, generate_latest,Gauge,CONTENT_TYPE_LATEST,start_http_server
from starlette.responses import Response

from base_class_api import BaseClass
from main import {{cookiecutter.model_name}}


app: FastAPI = FastAPI("Faster ML")

CPU_PERCENT = Gauge("cpu_percent", "CPU usage percentage")
CPU_FREQ = Gauge("cpu_frequency_mhz", "CPU frequency in MHz")
RAM_PERCENT = Gauge("ram_percent", "RAM usage percentage")
RAM_USED = Gauge("ram_used_bytes", "RAM used in bytes")
RAM_TOTAL = Gauge("ram_total_bytes", "Total RAM in bytes")

REQUEST_COUNT = Counter(
    "request_count", "App Request Count", ["method", "endpoint", "http_status"]
)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Request latency in seconds", ["endpoint"]
)

def update_cpu_metrics() -> None:
    CPU_PERCENT.set(psutil.cpu_percent())
    CPU_FREQ.set(psutil.cpu_freq().current)
    
def update_ram_metrics() -> None:
    ram = psutil.virtual_memory()
    RAM_PERCENT.set(ram.percent)
    RAM_USED.set(ram.used)
    RAM_TOTAL.set(ram.total)
    
def update_all_metrics() -> None:
    update_cpu_metrics()
    update_ram_metrics()


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
    return {"status": "Healthy"}


@app.get("/metrics")
async def metrics():
    """
    Expose Prometheus metrics at the /metrics endpoint.
    """
    update_all_metrics()
    
    return Response(generate_latest(), media_type="text/plain")


@app.post("/predict")
async def predict(data: BaseClass):
    """
    Dummy predict endpoint.
    Replace with your actual model prediction logic.
    """
    pre = {{cookiecutter.model_name}}()
    df = pd.DataFrame.from_records([data.__dict__])
    dx = pre.preprocess_data(df)

    with open(os.path.join(os.curdir, ".artifacts/model.pkl"), "rb") as file:
        model = pickle.load(file)

    cols = model.feature_names_in_

    dx = dx[cols]
    prediction = model.predict(dx)

    df["Survived"] = prediction
    conn = sqlite3.connect("data.db")
    df.to_sql("test_data", conn, if_exists="append")
    conn.close()

    return {"result": str(prediction[0])}
