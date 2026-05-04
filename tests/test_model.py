from pathlib import Path
import joblib
from ml.train import train_and_save

def test_train_creates_model_file(tmp_path: Path):
    model_file = tmp_path / "model.joblib"
    accuracy = train_and_save(model_path=model_file)
    assert model_file.exists(), "Файл моделі має бути створений"
    assert 0.0 <= accuracy <= 1.0, "Accuracy має бути коректним числом"
    assert accuracy > 0.8, f"Очікувано accuracy > 0.8, отримано {accuracy}"

def test_model_predicts_three_classes(tmp_path: Path):
    model_file = tmp_path / "model.joblib"
    train_and_save(model_path=model_file)
    model = joblib.load(model_file)
    sample = [[5.1, 3.5, 1.4, 0.2]]  # типова setosa
    pred = model.predict(sample)
    assert pred[0] in (0, 1, 2), "Клас має бути одним із 0/1/2"