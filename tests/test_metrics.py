from fastapi.testclient import TestClient
from ml.train import train_and_save
from app.main import app, MODEL_PATH, REFERENCE_PATH

if not MODEL_PATH.exists() or not REFERENCE_PATH.exists():
    train_and_save(MODEL_PATH, REFERENCE_PATH)

def test_metrics_endpoint_available():
    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "ml_predictions_total" in response.text
        assert "ml_prediction_latency_seconds" in response.text

def test_predict_increments_counter():
    with TestClient(app) as client:
        payload = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
        before = client.get("/metrics").text
        client.post("/predict", json=payload)
        client.post("/predict", json=payload)
        after = client.get("/metrics").text
        assert 'class_name="setosa",status="success"' in after
        assert before != after