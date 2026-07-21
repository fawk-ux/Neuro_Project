from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal
from typing import Any

class EEGInterpreter:
    """Класс для генерации клинических отчетов на основе классификации состояния."""

    @staticmethod
    def get_clinical_report(state: str) -> dict[str, Any]:
        reports = {
            'Relax/Neutral': {
                "status_ru": "Стабильное бодрствование (Релаксация / Нейтральное состояние)",
                "color": "normal",
                "summary": "Наблюдается нормализация альфа-ритма. Признаки психоэмоционального напряжения или утомления отсутствуют.",
                "recommendations": [
                    "Текущее функциональное состояние оптимально для выполнения аналитической работы.",
                    "Рекомендуется поддерживать текущий темп деятельности.",
                    "Проведение планового перерыва допускается через 60–90 минут."
                ]
            },
            'Concentration': {
                "status_ru": "Активное бодрствование (Высокая концентрация)",
                "color": "focus",
                "summary": "Выраженная депрессия альфа-ритма с доминированием бета-активности.",
                "recommendations": [
                    "Состояние характеризуется максимальной продуктивностью оператора.",
                    "Во избежание преждевременного истощения ресурсов внимания рекомендуется ограничить внешние раздражители.",
                    "Оптимальное время работы: 40–50 минут."
                ]
            },
            'Mental Fatigue': {
                "status_ru": "Выраженное умственное утомление (Астенизация)",
                "color": "fatigue",
                "summary": "Регистрируется дезорганизация основных корковых ритмов, увеличение медленноволновой активности.",
                "recommendations": [
                    "Рекомендована немедленная приостановка напряженной когнитивной деятельности.",
                    "Необходимо провести сеанс сенсорной разгрузки длительностью 15 минут.",
                    "Рекомендуется легкая физическая разминка."
                ]
            }
        }
        return reports.get(state, {
            "status_ru": "Неопределенный статус",
            "color": "neutral",
            "summary": "Недостаточно данных.",
            "recommendations": ["Проведите повторный сеанс."]
        })


class EEGRhythmAnalyzer:
    """Анализатор спектральной мощности ЭЭГ-ритмов."""

    RHYTHMS: dict[str, tuple[float, float]] = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, 45.0)
    }

    @staticmethod
    def bandpower(data: np.ndarray, fs: int, low: float, high: float) -> float:
        """Расчет спектральной мощности в заданном диапазоне (Welch method)."""
        if len(data) < 8:
            return 0.0
        nperseg = min(256, len(data))
        freqs, psd = signal.welch(data, fs, nperseg=nperseg, window='hann')
        idx = np.logical_and(freqs >= low, freqs <= high)
        if not np.any(idx):
            return 0.0
        if hasattr(np, 'trapezoid'):
            return float(getattr(np, 'trapezoid')(psd[idx], freqs[idx]))
        return float(getattr(np, 'trapz')(psd[idx], freqs[idx]))

    @classmethod
    def analyze_channels(
        cls,
        df: pd.DataFrame,
        fs: int = 128,
        channels: list[str] | None = None
    ) -> pd.DataFrame:
        """Расчет мощности ритмов для каждого канала."""
        if channels is None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            channels = [c for c in numerical_cols if c.lower() not in ['label', 'id', 'unnamed: 0']]

        results: dict[str, dict[str, float]] = {}
        for ch in channels:
            # Явное приведение к np.ndarray для Pylance
            ch_data: np.ndarray = np.asarray(df[ch].dropna())
            if len(ch_data) == 0:
                continue
            results[ch] = {}
            for name, (low, high) in cls.RHYTHMS.items():
                results[ch][name] = cls.bandpower(ch_data, fs, low, high)

        return pd.DataFrame(results).T

    @classmethod
    def get_mean_rhythms(
        cls,
        df: pd.DataFrame,
        fs: int = 128,
        channels: list[str] | None = None
    ) -> dict[str, float]:
        """Средние значения мощности ритмов по всем каналам (мкВ2/Гц)."""
        channel_df = cls.analyze_channels(df, fs, channels)
        if channel_df.empty:
            return {name: 0.0 for name in cls.RHYTHMS}
        # Явное приведение типа для Pylance
        mean_values: dict[str, float] = {}
        for col in channel_df.columns:
            mean_values[str(col)] = float(channel_df[col].mean())
        return mean_values

    @classmethod
    def get_rhythm_summary(
        cls,
        df: pd.DataFrame,
        fs: int = 128,
        channels: list[str] | None = None
    ) -> dict[str, Any]:
        """Расширенное резюме с максимальным ритмом и доминантой."""
        means = cls.get_mean_rhythms(df, fs, channels)
        if not means:
            return {"dominant_rhythm": "unknown", "dominant_value": 0.0, "mean_powers": means, "total_power": 0.0}

        dominant = max(means.keys(), key=lambda k: means[k])
        return {
            "dominant_rhythm": dominant,
            "dominant_value": round(means[dominant], 6),
            "mean_powers": {k: round(v, 6) for k, v in means.items()},
            "total_power": round(sum(means.values()), 6)
        }
