FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей (добавлен curl для healthcheck)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование и установка Python-зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Открытие портов
EXPOSE 8000
EXPOSE 8501

# Команда по умолчанию
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]