"""Тренування простої моделі класифікації Iris."""
from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"

def train_and_save(model_path: Path = MODEL_PATH) -> float:
    """Навчає модель та повертає точність на тестовій вибірці."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    joblib.dump(model, model_path)
    return accuracy

if __name__ == "__main__":
    acc = train_and_save()
    print(f"Model trained. Test accuracy: {acc:.4f}")
    print(f"Saved to: {MODEL_PATH}")