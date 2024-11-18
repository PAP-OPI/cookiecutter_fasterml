import time

from base_class_data import BaseClass
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

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
async def predict(data: BaseClass):
    """
    Dummy predict endpoint.
    Replace with your actual model prediction logic.
    """
    # Simulate processing time
    time.sleep(0.1)

    # Example prediction (Replace this with your model prediction)
    prediction = {"result": "default_prediction"}

    return prediction
