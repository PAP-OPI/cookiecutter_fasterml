import os
import pickle
import sys
import time
import sqlite3

#Memo test
import psutil
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.curdir))

import pandas as pd
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, generate_latest, Gauge, CollectorRegistry, Summary
from starlette.responses import Response
from typing import Dict, List

from base_class_api import BaseClass
from main import {{cookiecutter.model_name}}  

app: FastAPI = FastAPI()

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

MODEL_ACCURACY = Gauge("model_accuracy", "Model accuracy score")
MODEL_PRECISION = Gauge("model_precision", "Model precision score")
MODEL_RECALL = Gauge("model_recall", "Model recall score")
MODEL_F1 = Gauge("model_f1", "Model F1 score")
PREDICTIONS_COUNT = Counter("predictions_total", "Total number of predictions made")

def update_cpu_metrics() -> None:
    CPU_PERCENT.set(psutil.cpu_percent())
    CPU_FREQ.set(psutil.cpu_freq().current)
    
def update_ram_metrics() -> None:
    ram = psutil.virtual_memory()
    RAM_PERCENT.set(ram.percent)
    RAM_USED.set(ram.used)
    RAM_TOTAL.set(ram.total)
    
def calculate_model_metrics() -> Dict[str, float]:
    
    conn = sqlite3.connect("data.db")
    
    with open(os.path.join(os.curdir, ".artifacts/model.pkl"), "rb") as file:
        model = pickle.load(file)
        
    query:str = "" #TODO: Define the query to the database to get the data
    
    df = pd.read_sql_query(query, conn)
    
    if len(df) < 2:
        return {
            "accuracy":0,
            "precision":0,
            "recall":0,
            "f1_score":0
        }
        
    y_pred = df["PLACEHOLDER"].values #TODO: Define the way to read the predictions from the dataframe
    y_true = df["PLACEHOLDER"].values #TODO: Define the way to read he actual values from the dataframe
    
    return {
            "accuracy":accuracy_score(y_true=y_true, y_pred=y_pred),
            "precision":precision_score(y_true=y_true, y_pred=y_pred),
            "recall":recall_score(y_true=y_true, y_pred=y_pred),
            "f1_score":f1_score(y_true=y_true, y_pred=y_pred, zero_division=0)
        }
    
def update_model_metrics() -> None:
    
    metrics = calculate_model_metrics()
    
    MODEL_ACCURACY.set(metrics["accuracy"])
    MODEL_PRECISION.set(metrics["precision"])
    MODEL_RECALL.set(metrics["recall"])
    MODEL_F1.set(metrics["f1_score"])
    
def update_all_metrics() -> None:
    update_cpu_metrics()
    update_ram_metrics()
    update_model_metrics()
    
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
