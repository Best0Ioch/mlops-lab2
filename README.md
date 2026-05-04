# MLOps Lab 2: Iris ML API з CI/CD

---

## Опис проєкту

Даний проєкт реалізує наскрізний **MLOps-конвеєр** для задачі класифікації квіток Iris із розгортанням REST API та автоматизованим CI/CD процесом.

Мета роботи полягає у демонстрації повного життєвого циклу ML-системи: від підготовки даних і навчання моделі до її контейнеризації, тестування, безперервної інтеграції та деплойменту в хмарне середовище.

Задача класифікації полягає у визначенні виду квітки Iris (setosa, versicolor, virginica) на основі чотирьох числових ознак.

---

## Стек технологій

- Python 3.10+
- scikit-learn
- FastAPI
- Pydantic
- pytest
- Docker
- GitHub Actions
- Render

---

## Як запустити локально

### Клонування
```bash
git clone https://github.com/Best0Ioch/mlops-lab2.git
cd mlops-lab2
```

### Віртуальне середовище
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Залежності
```bash
pip install -r requirements.txt
```

### Навчання моделі
```bash
python -m ml.train
```

### Запуск API
```bash
uvicorn app.main:app --reload
```

---

## Запуск через Docker

### Збірка
```bash
docker build -t iris-ml-api .
```

### Запуск
```bash
docker run --rm -p 8000:8000 iris-ml-api
```

---

## Як запустити тести

```bash
pytest -q
```

---

## Як працює API

FastAPI сервіс містить такі ендпоінти:

- GET `/` — статус сервісу  
- GET `/health` — перевірка моделі  
- POST `/predict` — передбачення  

### Приклад запиту
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Приклад відповіді
```json
{
  "class_id": 0,
  "class_name": "setosa",
  "probability": 0.98
}
```

Swagger:
```
/docs
```

---

## Посилання на деплой

- https://mlops-lab2-np64.onrender.com
- https://mlops-lab2-np64.onrender.com/health
- https://mlops-lab2-np64.onrender.com/docs

---

## Структура репозиторію

```
mlops-lab2/
├── .github/workflows/ci.yml
├── app/
│   ├── main.py
│   ├── schemas.py
├── ml/
│   ├── train.py
├── tests/
│   ├── test_api.py
│   ├── test_model.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# Контрольні питання

## 1. CI vs CD
CI (Continuous Integration) — автоматичне тестування та інтеграція коду.  
CD (Continuous Deployment) — автоматичний деплой у продакшн.

---

## 2. Workflow, Job, Step
- Workflow — повний CI/CD процес  
- Job — набір задач на одному runner  
- Step — окрема дія (команда або action)

Ієрархія:
Workflow → Job → Step

---

## 3. startup vs predict
Завантаження моделі в `@app.on_event("startup")` дозволяє:
- завантажити модель 1 раз
- уникнути перевантаження при кожному запиті

Якщо завантажувати в predict:
- різке падіння продуктивності
- збільшення latency
- зайве IO навантаження

---

## 4. Docker cache
Спочатку копіюється `requirements.txt`, щоб:
- кешувати встановлення залежностей

Якщо змінити тільки код:
- Docker не перевстановлює залежності
- значно швидший rebuild

---

## 5. Pydantic validation
FastAPI використовує Pydantic схему `IrisFeatures`.

Якщо передати рядок замість числа:
- виникає помилка валідації
- повертається HTTP 422 Unprocessable Entity

Це забезпечує:
- типобезпеку
- захист API
- раннє виявлення помилок
