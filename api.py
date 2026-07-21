from __future__ import annotations

import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Union

app = FastAPI(
    title="EEG Brainwave API",
    description="Backend server for EEG mental state classification. Integrated with trained ML model for ITMO Stars.",
    version="2.2",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Пути к сохраненным артефактам
MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"
META_PATH = "models/model_meta.json"

# Глобальные переменные для модели
model = None
scaler = None
meta = {}
FEATURE_NAMES: List[str] = []
TARGET_NAMES: Dict[str, str] = {"0": "Relax/Neutral", "1": "Concentration", "2": "Mental Fatigue"}


@app.on_event("startup")
def load_artifacts():
    global model, scaler, meta, FEATURE_NAMES, TARGET_NAMES
    
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Loaded ML model from {MODEL_PATH}")
    else:
        print(f"⚠️ Model not found at {MODEL_PATH}. Run neuro_project_v2.py first.")

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print(f"✅ Loaded Scaler from {SCALER_PATH}")

    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
            FEATURE_NAMES = meta.get("feature_names", [])
            TARGET_NAMES = meta.get("target_names", TARGET_NAMES)
            print(f"✅ Loaded Metadata ({len(FEATURE_NAMES)} features expected)")


class EEGFeaturesInput(BaseModel):
    features: Union[Dict[str, float], List[float]] = Field(
        ..., 
        description="Словарь названий фич и значений или вектор фич"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": {
                        "AF3_alpha": 12.45,
                        "AF3_beta": 8.12,
                        "F3_theta": 15.30,
                        "alpha_beta_ratio": 1.53,
                        "spectral_entropy": 0.84
                    }
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    class_index: int
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_used: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "class_index": 1,
                    "prediction": "Concentration",
                    "confidence": 0.9125,
                    "probabilities": {
                        "Relax/Neutral": 0.0521,
                        "Concentration": 0.9125,
                        "Mental Fatigue": 0.0354
                    },
                    "model_used": "XGBoost"
                }
            ]
        }
    }


@app.get("/")
def home() -> dict[str, Any]:
    return {
        "status": "active" if model is not None else "degraded (model missing)",
        "project": "ITMO Stars EEG Analysis",
        "model_loaded": meta.get("best_model", "None"),
        "cv_accuracy": meta.get("cv_accuracy", "N/A"),
        "expected_features_count": len(FEATURE_NAMES),
        "endpoints": ["POST /predict"]
    }


@app.post("/predict", response_model=PredictResponse)
def predict(data: EEGFeaturesInput) -> dict[str, Any]:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="ML Model is not loaded on server. Run neuro_project_v2.py to train and save model.pkl"
        )

    try:
        # Преобразование входных данных в DataFrame
        if isinstance(data.features, dict):
            cols = FEATURE_NAMES if FEATURE_NAMES else list(data.features.keys())
            input_dict = {col: [data.features.get(col, 0.0)] for col in cols}
            df_input = pd.DataFrame(input_dict)
        elif isinstance(data.features, list):
            if FEATURE_NAMES and len(data.features) != len(FEATURE_NAMES):
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected {len(FEATURE_NAMES)} features, but got {len(data.features)}."
                )
            cols = FEATURE_NAMES if len(FEATURE_NAMES) == len(data.features) else None
            df_input = pd.DataFrame([data.features], columns=cols)
        else:
            raise HTTPException(status_code=400, detail="Invalid feature format.")

        # Замена возможного NaN / Inf
        df_input = df_input.fillna(0.0).replace([np.inf, -np.inf], 0.0)

        # Масштабирование признаков (если модель требует scaled инпут)
        if meta.get("best_model") == "LogisticRegression" and scaler is not None:
            X_val = scaler.transform(df_input)
        else:
            X_val = df_input

        # Честный инференс обученной модели
        pred_idx = int(model.predict(X_val)[0])
        state_name = TARGET_NAMES.get(str(pred_idx), f"Class_{pred_idx}")

        # Вычисление реальной уверенности и вероятностей классов
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val)[0]
            confidence = float(np.max(probs))
            prob_dict = {
                TARGET_NAMES.get(str(i), f"Class_{i}"): round(float(p), 4) 
                for i, p in enumerate(probs)
            }
        else:
            confidence = 1.0
            prob_dict = {state_name: 1.0}

        return {
            "class_index": pred_idx,
            "prediction": state_name,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "model_used": meta.get("best_model", "Unknown")
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")