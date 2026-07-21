Классические модели дают ~72% accuracy. Проблема в том, что фичи — это агрегаты, а ЭЭГ имеет пространственно-временную структуру, которую деревья не улавливают.
Решение — EEG-2D-CNN:

    Вход: матрица 4×N (каналы × временные отсчёты)
    Conv2D извлекает пространственные паттерны (синхронизация полушарий)
    Conv1D по времени — динамику ритмов

Python

import torch
import torch.nn as nn

class EEG2DCNN(nn.Module):
    def __init__(self, n_channels=4, n_classes=3):
        super().__init__()
        self.spatial = nn.Conv2d(1, 32, (n_channels, 1))  # Пространственный фильтр
        self.temporal = nn.Conv1d(32, 64, kernel_size=16)  # Временной паттерн
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 245, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.spatial(x)      # [batch, 32, 1, time]
        x = x.squeeze(2)         # [batch, 32, time]
        x = self.temporal(x)     # [batch, 64, time-15]
        return self.classifier(x)

3. Главная ловушка: data leakage
Стандартный train_test_split даст завышенную точность. Почему? Записи одного испытуемого коррелированы — модель запомнит человека, а не состояние.
Решение — Subject-wise GroupKFold:
Python

from sklearn.model_selection import GroupKFold

# Группы по испытуемым
groups = df['subject_id']  # 4 уникальных значения
cv = GroupKFold(n_splits=4)

for train_idx, test_idx in cv.split(X, y, groups):
    # В тесте — один испытуемый, которого модель не видела
    ...

Результат: accuracy падает с 89% до 81%, но это реальная точность.
4. Спектральный анализ: Welch + ритмы
Помимо классификации, важно понимать почему модель приняла такое решение. Спектральная мощность по диапазонам:
Table
Ритм	Частота	Состояние
δ (дельта)	0.5–4 Гц	Глубокий сон, кома
θ (тета)	4–8 Гц	Сонность, медитация
α (альфа)	8–13 Гц	Релаксация, закрытые глаза
β (бета)	13–30 Гц	Концентрация, активное мышление
γ (гамма)	30–45 Гц	Высокоуровневое познание
Python

from scipy import signal

def bandpower(data, fs, band):
    fmin, fmax = band
    freqs, psd = signal.welch(data, fs, nperseg=256)
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return np.trapz(psd[idx], freqs[idx])

# Применяем ко всем каналам
rhythms = {name: bandpower(ch, 128, band) 
           for name, band in [('alpha', (8,13)), ('beta', (13,30)), ...]}

5. Интерфейс: FastAPI + Streamlit
Чтобы исследователи без Python могли пользоваться системой, собрали полноценный продукт:
Backend (FastAPI):
Python

@app.post("/predict")
def predict(data: EEGFeatures):
    feats = np.array(data.features)
    score = np.mean(feats) * np.std(feats)
    # Пороговая логика для 3 классов
    ...

@app.post("/rhythms")
def calculate_rhythms(req: EEGRhythmRequest):
    # Welch PSD по каналам
    ...

Frontend (Streamlit):

    Загрузка CSV
    Визуализация осциллограмм
    Классификация с клиническим заключением
    Тепловая карта ритмов

 Дашборд 
6. Explainability: SHAP + LLM
Модель — чёрный ящик. Для клинического применения нужно объяснение:
Python

import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Генерация текста через LLM
summary = f"""
Доминантный ритм: {dominant_rhythm}.
Альфа-активность снижена на {alpha_drop}% — 
признак когнитивного напряжения.
"""

Результаты
Table
Метрика	Random Forest	EEG-2D-CNN
Accuracy	72.3%	81.4%
F1-macro	0.71	0.80
Inference	2 мс	5 мс
Что дальше

    Emotiv EPOC — 14 каналов вместо 4, точность 90%+
    Адаптивное обучение — корректировка темпа лекций по ЭЭГ
    Real-time pipeline — обработка потока 250 Гц онлайн

* **GitHub:** [github.com/fawk-ux/neuro-project](https://github.com/fawk-ux/neuro-project)
* **Проект выполнен в рамках программы STARS, Университет ИТМО.**
