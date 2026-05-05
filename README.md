# MLOps Lab 3: Моніторинг ML API та детекція проблем

[![CI/CD Pipeline](https://github.com/Best0Ioch/mlops-lab2/actions/workflows/ci.yml/badge.svg)](https://github.com/Best0Ioch/mlops-lab2/actions/workflows/ci.yml)

---

# 1. Опис проєкту

Метою роботи є розширення ML API з ЛР2 системою моніторингу на основі Prometheus, детектором data drift на основі KS-тесту та структурованим JSON-логуванням.

Задача: забезпечити спостережуваність ML-сервісу для своєчасного виявлення деградації моделі через зміну вхідних даних.

У роботі реалізовано:

- Prometheus-метрики (Counter, Histogram, Gauge)
- ендпоінт /metrics у форматі Prometheus exposition format
- ендпоінт /check-drift для перевірки зсуву даних
- клас DriftDetector на основі KS-тесту
- структуроване JSON-логування
- docker-compose з Prometheus
- Evidently HTML-звіт про drift

---

# 2. Реалізовані метрики

| Метрика | Тип | Мітки | Опис |
|---------|-----|-------|------|
| ml_predictions_total | Counter | class_name, status | Загальна кількість прогнозів моделі |
| ml_prediction_latency_seconds | Histogram | - | Час обробки одного прогнозу (секунди) |
| ml_prediction_confidence | Histogram | - | Розподіл значень predict_proba |
| ml_errors_total | Counter | error_type | Кількість помилкових запитів |
| ml_model_loaded | Gauge | - | Статус завантаження моделі (1/0) |
| ml_drift_checks_total | Counter | - | Загальна кількість перевірок на drift |
| ml_drift_detected_total | Counter | feature | Кількість випадків виявленого drift |

---

# 3. Drift detection

Реалізовано клас `DriftDetector` на основі двовибіркового KS-тесту Колмогорова-Смирнова (`scipy.stats.ks_2samp`).

Reference-вибірка (тренувальні дані) зберігається у файлі `reference_stats.joblib`.

Ендпоінт `POST /check-drift` приймає батч із мінімум 10 зразків (кожен із 4 ознак) та параметр `alpha` (поріг значущості, за замовчуванням 0.05). Повертає результат із прапором `drift_detected` для кожної ознаки та загальним висновком.

**Результати перевірки:**

- Нормальні дані (типові значення Iris): `drift_detected: false`, `n_drifted_features: 0`
- Аномальні дані (зсув на +3–4 см): `drift_detected: true`, `n_drifted_features: 4`, p_value ≈ 0 для всіх ознак

---

# 4. Evidently звіт

Додатково згенеровано HTML-звіт за допомогою бібліотеки Evidently.

**Результати Evidently (drift_report.html):**

- Загальний висновок: Dataset Drift is NOT detected
- Drifted Columns: 1 з 4 (25%)
- Drift Score для кожної ознаки:

| Ознака | Drift Score | Результат |
|--------|-------------|-----------|
| sepal_width | 0.999959 | Not Detected |
| sepal_length | 0.994547 | Not Detected |
| petal_width | 0.982743 | Not Detected |
| petal_length | 0 | **Detected** |

Drift виявлено тільки в `petal_length`, оскільки в скрипті було штучно додано зсув +1.5 саме для цієї ознаки. Решта ознак залишилися без змін, що підтверджує коректність детекції.

---

# 5. Логування

Структуроване JSON-логування реалізовано через бібліотеку `python-json-logger`.

Ключові події, що логуються:

- `startup` — старт сервісу, статус завантаження моделі та drift detector
- `prediction` — кожен прогноз із класом, ймовірністю та вхідними ознаками
- `drift_check` — результат перевірки drift із кількістю зразків та переліком ознак зі зсувом
- `inference_error` — помилки інференсу

Приклад логу:

```json
{"timestamp": "2026-05-04T10:33:15Z", "level": "INFO", "logger": "ml-api", "event": "prediction", "class_id": 0, "class_name": "setosa", "probability": 0.9812}
6. Приклади запитів
/predict

POST /predict
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
Відповідь:

{
    "class_id": 0,
    "class_name": "setosa",
    "probability": 0.9812
}
/check-drift (нормальні дані)
POST /check-drift
{
    "samples": [
        [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],
        [5.4, 3.9, 1.7, 0.4], [5.0, 3.6, 1.4, 0.2], [5.5, 2.5, 4.0, 1.3],
        [6.1, 2.9, 4.7, 1.4], [6.0, 3.0, 4.8, 1.8], [6.3, 2.5, 5.0, 1.9],
        [6.5, 3.0, 5.2, 2.0]
    ],
    "alpha": 0.05
}
Відповідь:

{
    "drift_detected": false,
    "n_drifted_features": 0,
    "drifted_features": [],
    "per_feature": {
        "sepal_length": {"statistic": 0.2, "p_value": 0.5, "drift_detected": false},
        "sepal_width": {"statistic": 0.15, "p_value": 0.7, "drift_detected": false},
        "petal_length": {"statistic": 0.25, "p_value": 0.3, "drift_detected": false},
        "petal_width": {"statistic": 0.18, "p_value": 0.6, "drift_detected": false}
    },
    "n_samples": 10,
    "alpha": 0.05
}
/check-drift (drift дані)
POST /check-drift
{
    "samples": [
        [9.0, 8.0, 8.0, 5.0], [9.5, 7.5, 8.5, 5.5], [8.5, 8.5, 7.5, 4.5],
        [9.2, 8.2, 8.2, 5.2], [9.8, 7.8, 8.8, 5.8], [8.8, 8.8, 7.8, 4.8],
        [9.4, 8.4, 8.4, 5.4], [9.6, 7.6, 8.6, 5.6], [8.6, 8.6, 7.6, 4.6],
        [9.1, 8.1, 8.1, 5.1], [9.3, 8.3, 8.3, 5.3], [9.7, 7.7, 8.7, 5.7]
    ],
    "alpha": 0.05
}
Відповідь:

{
    "drift_detected": true,
    "n_drifted_features": 4,
    "drifted_features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
    "per_feature": {
        "sepal_length": {"statistic": 1.0, "p_value": 0.0, "drift_detected": true},
        "sepal_width": {"statistic": 1.0, "p_value": 0.0, "drift_detected": true},
        "petal_length": {"statistic": 1.0, "p_value": 0.0, "drift_detected": true},
        "petal_width": {"statistic": 1.0, "p_value": 0.0, "drift_detected": true}
    },
    "n_samples": 12,
    "alpha": 0.05
}
```
7. Як запустити моніторинг

cd monitoring
docker-compose -f docker-compose.monitoring.yml up --build
Prometheus: http://localhost:9090
ML API: http://localhost:8000

Корисні PromQL запити:

rate(ml_predictions_total[1m]) — швидкість прогнозів за секунду

histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket[5m])) — 95-й перцентиль latency

sum by (class_name) (ml_predictions_total) — розподіл прогнозів за класами

ml_drift_detected_total — кількість виявлених drift

8. Як запустити тести

pytest -q
10 тестів мають пройти успішно.

9. Структура проєкту

mlops-lab2/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   ├── metrics.py
│   ├── drift.py
│   └── logging_config.py
├── ml/
│   ├── __init__.py
│   └── train.py
├── tests/
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_api.py
│   ├── test_metrics.py
│   └── test_drift.py
├── monitoring/
│   ├── prometheus.yml
│   └── docker-compose.monitoring.yml
├── scripts/
│   └── evidently_report.py
├── Dockerfile
├── requirements.txt
└── README.md
10. Висновки
Реалізовано повноцінний observable ML-сервіс із метриками Prometheus, детекцією data drift через KS-тест та структурованим логуванням. Система дозволяє виявляти деградацію моделі до того, як вона вплине на бізнес-показники.

Evidently звіт підтвердив коректність роботи детектора: drift виявлено лише в ознаці зі штучним зсувом (petal_length), інші три ознаки — без змін.

11. Контрольні питання
1. Різниця між моніторингом класичного веб-сервісу та ML-сервісу
Класичний веб-сервіс моніториться за технічними метриками: latency, throughput, error rate, CPU/RAM. Цього достатньо, бо поведінка сервісу змінюється лише при зміні коду.

ML-сервіс потребує додаткових метрик: розподіл вхідних ознак, упевненість моделі (confidence), розподіл прогнозів за класами, метрики якості (accuracy, precision, recall). Причина в тому, що модель може деградувати без жодних змін у коді — лише через те, що вхідні дані почали відрізнятися від тренувальних (data drift). Стандартні метрики не покажуть проблему: сервіс продовжує повертати 200 OK, але якість прогнозів падає.

2. Pull-модель Prometheus
Prometheus сам ініціює збір метрик (scraping) — це pull-модель. Він періодично робить HTTP-запити до /metrics кожного target.

Якщо ML API повертає 500 на /metrics або стає недоступним, Prometheus позначає ціль як DOWN у своєму інтерфейсі (/targets). Scrape продовжується за розкладом. Коли сервіс відновлюється, статус змінюється на UP. Це дозволяє швидко виявити проблему без додаткових алертів.

3. Різниця між Counter, Gauge та Histogram
Counter — метрика, що тільки зростає (або обнуляється при перезапуску). Використовується для підрахунку кількості подій: кількість запитів, кількість помилок. У PromQL аналізується через rate().

Gauge — метрика, що може зростати і спадати. Відображає поточний стан: температура, використання пам'яті, кількість активних з'єднань.

Histogram — метрика, що збирає розподіл значень за кошиками (buckets). Складається з _bucket, _sum, _count. Дозволяє обчислювати перцентилі (p50, p95, p99) через histogram_quantile(). Ідеально підходить для latency.

4. Data drift та його наслідки
Data drift — зміна статистичного розподілу вхідних даних у продакшені порівняно з даними, на яких модель навчалася.

Три причини виникнення drift на прикладі моделі оцінки кредитного ризику:

Економічні зміни: інфляція призвела до зростання середньої зарплати з 15 000 до 25 000 грн. Модель, навчена на старих даних, недооцінює ризик для клієнтів із вищими доходами.

Зміна джерела даних: банк змінив CRM-систему, і тепер поле "стаж роботи" заповнюється в місяцях замість років. Модель отримує значення 120 замість 10, що спотворює прогнози.

Розширення аудиторії: сервіс запустили в новому регіоні з іншою демографією. Розподіл віку, доходу та кредитної історії нових клієнтів відрізняється від тренувальної вибірки.

5. KS-тест та p-value
KS-тест (Kolmogorov-Smirnov) — непараметричний статистичний тест, що порівнює дві вибірки та перевіряє гіпотезу про те, що вони походять з одного розподілу.

Тест обчислює статистику D — максимальну відстань між двома емпіричними кумулятивними функціями розподілу (CDF). Чим більше D, тим сильніше відрізняються вибірки.

p-value — імовірність побачити таке ж або більше значення D за умови, що нульова гіпотеза (розподіли однакові) вірна. p-value НЕ є ймовірністю того, що вибірки різні.

Інтерпретація:

p-value < 0.05 → відкидаємо нульову гіпотезу, розподіли статистично значуще відрізняються → drift detected

p-value ≥ 0.05 → немає підстав відкидати нульову гіпотезу → drift not detected
