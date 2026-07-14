🧠 EEG Brainwave Dataset: Mental State Analysis

    ITMO Stars Research Project — программный комплекс для анализа ЭЭГ-активности и классификации ментального состояния оператора.

https://python.org
https://fastapi.tiangolo.com
https://streamlit.io
https://pytorch.org
📋 Описание
Система предназначена для мониторинга функционального состояния операторов по данным электроэнцефалографии (ЭЭГ). Реализована в рамках научно-исследовательской работы Университета ИТМО.
Архитектура
Table
Компонент	Технология	Назначение
Backend	FastAPI + Uvicorn	REST API для классификации и спектрального анализа
Frontend	Streamlit	Интерактивный дашборд с визуализацией
ML Core	PyTorch + scikit-learn	EEG-2D-CNN + классические модели
Explainability	SHAP + LLM	Интерпретация решений модели
Validation	Subject-wise GroupKFold	Научная честность, без утечки данных
🚀 Быстрый старт
Требования

    Python 3.10+
    4 GB RAM (рекомендуется 8 GB для CNN)

Установка
bash

# Клонирование репозитория
git clone https://github.com/fawk-ux/neuro-project.git
cd neuro-project

# Создание виртуального окружения
python -m venv .venv

# Windows
.venv\Scripts\activate.bat

# Linux/macOS
source .venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

Запуск
bash

# Вариант 1: через скрипт (рекомендуется)
start.bat

# Вариант 2: ручной запуск
# Терминал 1 — API
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Терминал 2 — Dashboard
streamlit run dashboard.py

Откройте в браузере:

    Dashboard: http://localhost:8501
    API Docs: http://localhost:8000/docs

📊 Возможности
1. Визуализация ЭЭГ-сигналов

    Осциллограмма многоканальной записи
    Группировка статистических признаков по типу
    Расшифровка обозначений (lag1_mean, q1, d_h2h1 и др.)

2. Классификация состояния
Table
Класс	Описание	Показатели
Relax/Neutral	Стабильное бодрствование	Нормализация α-ритма
Concentration	Высокая концентрация	Депрессия α, доминирование β
Mental Fatigue	Умственное утомление	Дезорганизация ритмов, рост медленных волн

    REST API endpoint /predict
    Клиническое заключение с рекомендациями
    Уверенность модели (confidence score)

3. Спектральный анализ

    Расчёт мощности ритмов: δ (0.5–4 Гц), θ (4–8 Гц), α (8–13 Гц), β (13–30 Гц), γ (30–45 Гц)
    Метод Уэлча (Welch's periodogram)
    Тепловая карта «каналы × диапазоны»
    Доминантный ритм и общая спектральная мощность

🏗️ Структура проекта
plain

neuro-project/
├── 📁 .streamlit/           # Конфигурация темы
│   └── config.toml
├── 📁 .vscode/              # Настройки VS Code
│   └── settings.json
├── 📄 analytics.py          # Аналитический модуль (ритмы + интерпретатор)
├── 📄 api.py                # FastAPI backend (/predict, /rhythms)
├── 📄 dashboard.py          # Streamlit frontend
├── 📄 neuro_project.py      # Базовый ML-скрипт
├── 📄 neuro_project_v2.py   # Расширенная версия
├── 📄 requirements.txt      # Зависимости
├── 📄 start.bat             # Скрипт запуска (Windows)
├── 📄 mental-state.csv      # Демо-датасет
└── 📄 README.md             # Вы здесь

🧪 Датасет
Используется обработанный датасет EEG brainwave dataset: mental state (Jordan J. Bird et al., IEEE 2018).

    4 испытуемых (2 мужчины, 2 женщины)
    3 состояния: relaxed, concentrating, neutral
    60 секунд записи на состояние
    4 канала: TP9, AF7, AF8, TP10 (Muse EEG headband)
    Статистические фичи: mean, std, quartiles, inter-hemispheric differences

🐳 Docker
bash

# Сборка образа
docker-compose build

# Запуск
docker-compose up

# Дашборд: http://localhost:8501
# API: http://localhost:8000

🔬 Научная составляющая
EEG-2D-CNN
Архитектура свёрточной нейросети для анализа пространственно-временной структуры ЭЭГ:

    Вход: 2D-матрица «каналы × время»
    Слои: Conv2D → BatchNorm → MaxPool → Dropout → FC
    Выход: 3 класса состояния

Subject-wise GroupKFold

    Разделение по испытуемым, а не по записям
    Исключает data leakage между тренировкой и тестом
    Критично для биомедицинских данных

SHAP + LLM Explanations

    SHAP: локальная интерпретация вклада каждого признака
    LLM: генерация клинического резюме на естественном языке

📹 Демо
Видео-демонстрация работы дашборда (2 мин)
📝 Публикации

    Habr: Разбор EEG-2D-CNN для классификации ментального состояния (в процессе)

🔮 Roadmap

    [ ] Интеграция с Emotiv EPOC (живые данные)
    [ ] Адаптивная система обучения для студентов ИТМО
    [ ] Расширение до 5+ классов состояний
    [ ] Мобильное приложение для оперативного мониторинга

👤 Автор
Университет ИТМО — научно-исследовательская работа в рамках программы STARS.
📄 Лицензия
MIT License — свободное использование для исследовательских и образовательных целей.

    Примечание: Данное ПО предназначено исключительно для исследовательских целей. Не используйте для медицинской диагностики без сертификации.