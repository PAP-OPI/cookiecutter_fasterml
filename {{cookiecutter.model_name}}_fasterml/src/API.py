import os
import pickle
import sys
import time
import sqlite3

#Memo test
import psutil
import numpy as np
from scipy.stats import ks_2samp

sys.path.append(os.path.abspath(os.curdir))

import pandas as pd
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, generate_latest, Gauge, CollectorRegistry, Summary
from starlette.responses import Response
from typing import Dict, List

from base_class_api import BaseClass
from main import {{cookiecutter.model_name}}  

app: FastAPI = FastAPI()

KS = Histogram("ks_test", "pvalues")

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
    """Function to update CPU usage metrics
    """
    CPU_PERCENT.set(psutil.cpu_percent())
    CPU_FREQ.set(psutil.cpu_freq().current)
    
#TODO: Implement a way to monitor GPU usage and develop their respective metrics
    
def update_ram_metrics() -> None:
    """Function to update RAM usage metrics
    """
    ram = psutil.virtual_memory()
    RAM_PERCENT.set(ram.percent)
    RAM_USED.set(ram.used)
    RAM_TOTAL.set(ram.total)
    
#TODO: Add data drift metrics support for prometheus
'''
def calculate_model_metrics() -> Dict[str, float]:
    """Function to calculate standard statistical tests for telemetry

    Returns:
        Dict[str, float]: Dictionary with the name of the metric as a key, and the value is its respective value.
    """
    ks_pvalues: dict = {}
    
    conn = sqlite3.connect("data.db")
    
    with open(os.path.join(os.curdir, ".artifacts/model.pkl"), "rb") as file:
        model = pickle.load(file)
        
    query_train :str = "SELECT * FROM train_data" #TODO: Define the query to the database to get the data
    query_prod:str = "SELECT * FROM production_data"
    
    train_sample = pd.read_sql_query(query_train, conn)
    new_sample= pd.read_sql_query(query_prod, conn)
    
    conn.close()
    
    for column in new_sample.columns:
        try:
            test = ks_2samp(new_sample[column].values, train_sample[column].values)
            ks_pvalues.update({"ks_value":})
        except:
            #TODO: Add FasterML exception as warning
            pass
    
    return {"ks_value":test.pvalue(), "column": }
'''

def update_model_metrics() -> None:
    """Function to update model metrics
    """
    pass
    
    
def update_all_metrics() -> None:
    """Function to update all of the metrics for prometheus integration
    """
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
    calculate_model_metrics()
    
    return Response(generate_latest(), media_type="text/plain")


@app.post("/predict")
async def predict(data: BaseClass):
    """
    Predict endpoint.
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
    df.to_sql("production_data", conn, if_exists="append")
    conn.close()

    return {"result": str(prediction[0])}
