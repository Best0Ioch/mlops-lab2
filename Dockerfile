# Базовий образ – мінімальний Python для зменшення розміру
FROM python:3.11-slim

# Системні налаштування
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Спочатку копіюємо лише requirements - щоб Docker кешував шар із залежностями
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Копіюємо решту коду
COPY app ./app
COPY ml ./ml

# Тренуємо модель прямо при збірці образу
RUN python -m ml.train

# Render передає порт через змінну $PORT, локально використовуємо 8000
ENV PORT=8000
EXPOSE 8000

# Запуск Uvicorn з прив'язкою до 0.0.0.0
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]