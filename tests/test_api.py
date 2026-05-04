from fastapi.testclient import TestClient
from ml.train import train_and_save
from app.main import app, MODEL_PATH

# Гарантуємо існування файлу моделі перед запуском API-тестів
if not MODEL_PATH.exists():
    train_and_save(MODEL_PATH)

def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True

def test_predict_setosa():
    with TestClient(app) as client:
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["class_name"] == "setosa"
        assert 0.0 <= body["probability"] <= 1.0

def test_predict_invalid_input():
    with TestClient(app) as client:
        payload = {"sepal_length": "not-a-number"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Pydantic validation error