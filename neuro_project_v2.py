"""
NEURO PROJECT v2.0 — ITMO STARS EDITION
EEG Mental State Classification with Deep Learning & Model Export

Requirements:
    pip install pandas numpy matplotlib seaborn scikit-learn torch shap xgboost joblib

Usage:
    python neuro_project_v2.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json
import joblib

from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, cohen_kappa_score
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# Optional imports
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[WARN] XGBoost not installed. Install: pip install xgboost")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not installed. Install: pip install torch")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] SHAP not installed. Install: pip install shap")

# ============================
# CONFIGURATION
# ============================
DATA_PATH = 'mental-state.csv'
TARGET_COL = 'Label'
SUBJECT_COL = None 
RANDOM_STATE = 42
N_SPLITS = 5
DEVICE = torch.device('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')

TARGET_NAMES = {0: 'Relax/Neutral', 1: 'Concentration', 2: 'Mental Fatigue'}

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 9

# ============================
# 1. LOAD DATA
# ============================
print("=" * 70)
print("NEURO PROJECT v2.0 — ITMO STARS EDITION")
print("=" * 70)

print(f"\n[1] Loading data from {DATA_PATH}...")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"'{DATA_PATH}' not found. Please place your mental-state.csv in the same folder."
    )

df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# Auto-detect subject column
subject_candidates = [c for c in df.columns if c.lower() in [
    'subject', 'subject_id', 'user', 'user_id', 'participant', 'id', 'subjectid', 'subj'
]]
if subject_candidates:
    SUBJECT_COL = subject_candidates[0]
    print(f"Auto-detected subject column: '{SUBJECT_COL}'")
else:
    print("No subject column detected. Using standard StratifiedKFold.")

# Identify feature columns
exclude_cols = [TARGET_COL, 'State_Name', SUBJECT_COL] if SUBJECT_COL else [TARGET_COL, 'State_Name']
feature_cols = [c for c in df.columns if c not in exclude_cols and c != TARGET_COL]
print(f"Feature columns: {len(feature_cols)}")

X = df[feature_cols].copy()
y = df[TARGET_COL].astype(int).copy()
groups = df[SUBJECT_COL].copy() if SUBJECT_COL else None

# Ensure numeric
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.median())

print(f"\nTarget distribution:")
print(y.value_counts().sort_index())

# ============================
# 2. EDA & VISUALIZATIONS
# ============================
print("\n[2] Generating EDA visualizations...")
os.makedirs('output', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Correlation heatmap
top_var_features = X.var().nlargest(20).index.tolist()
corr = X[top_var_features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', center=0, ax=axes[0,0], square=True)
axes[0,0].set_title('Top 20 Features Correlation Matrix', fontweight='bold')

# PCA 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
scatter = axes[0,1].scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis', alpha=0.6, edgecolors='k', linewidth=0.3)
axes[0,1].set_title(f'PCA 2D (explained var: {pca.explained_variance_ratio_.sum()*100:.1f}%)', fontweight='bold')
legend = axes[0,1].legend(*scatter.legend_elements(), title="Class")
axes[0,1].add_artist(legend)

# t-SNE
if len(X) > 500:
    idx = np.random.choice(len(X), 500, replace=False)
    X_tsne_sample, y_tsne_sample = X.iloc[idx], y.iloc[idx]
else:
    X_tsne_sample, y_tsne_sample = X, y

tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=min(30, len(X_tsne_sample)-1))
X_tsne = tsne.fit_transform(StandardScaler().fit_transform(X_tsne_sample))
scatter2 = axes[1,0].scatter(X_tsne[:,0], X_tsne[:,1], c=y_tsne_sample, cmap='viridis', alpha=0.7, edgecolors='k', linewidth=0.3)
axes[1,0].set_title('t-SNE 2D (sample=500)', fontweight='bold')
legend2 = axes[1,0].legend(*scatter2.legend_elements(), title="Class")
axes[1,0].add_artist(legend2)

# Feature variance
variance = X.var().sort_values(ascending=False)
axes[1,1].bar(range(min(20, len(variance))), variance.values[:20], color='steelblue', edgecolor='black')
axes[1,1].set_xticks(range(min(20, len(variance))))
axes[1,1].set_xticklabels(variance.index[:20], rotation=45, ha='right', fontsize=7)
axes[1,1].set_title('Top 20 Feature Variances', fontweight='bold')
axes[1,1].set_ylabel('Variance')

plt.tight_layout()
plt.savefig('output/01_eda_overview.png', bbox_inches='tight')
plt.close()

# ============================
# 3. FEATURE ENGINEERING
# ============================
print("\n[3] Feature Engineering...")
X_fe = X.copy()

# Spectral ratios
band_cols = {}
for band in ['alpha', 'beta', 'theta', 'gamma', 'delta']:
    band_cols[band] = [c for c in feature_cols if f'_{band}' in c.lower()]

if band_cols['alpha'] and band_cols['beta']:
    X_fe['alpha_beta_ratio'] = X_fe[band_cols['alpha']].mean(axis=1) / (X_fe[band_cols['beta']].mean(axis=1) + 1e-8)
if band_cols['theta'] and band_cols['beta']:
    X_fe['theta_beta_ratio'] = X_fe[band_cols['theta']].mean(axis=1) / (X_fe[band_cols['beta']].mean(axis=1) + 1e-8)
if band_cols['alpha'] and band_cols['theta']:
    X_fe['alpha_theta_ratio'] = X_fe[band_cols['alpha']].mean(axis=1) / (X_fe[band_cols['theta']].mean(axis=1) + 1e-8)

if band_cols['beta'] and band_cols['alpha'] and band_cols['theta']:
    X_fe['engagement_index'] = X_fe[band_cols['beta']].mean(axis=1) / (
        X_fe[band_cols['alpha']].mean(axis=1) + X_fe[band_cols['theta']].mean(axis=1) + 1e-8
    )

# Безопасный расчет спектральной энтропии (без NaN)
X_pos = np.maximum(X_fe[feature_cols].values, 0)
sums = X_pos.sum(axis=1, keepdims=True) + 1e-8
p = X_pos / sums
X_fe['spectral_entropy'] = -np.sum(p * np.log(p + 1e-8), axis=1)

print(f"  Features after engineering: {X_fe.shape[1]} (was {X.shape[1]})")

# ============================
# 4. CV SETUP
# ============================
print("\n[4] Setting up Cross-Validation...")
if groups is not None and groups.nunique() > 5:
    try:
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        print(f"  Using StratifiedGroupKFold (subjects={groups.nunique()})")
        cv_strategy = 'group'
    except Exception:
        cv = GroupKFold(n_splits=N_SPLITS)
        print(f"  Using GroupKFold (subjects={groups.nunique()})")
        cv_strategy = 'group'
else:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    print(f"  Using StratifiedKFold (no subject info)")
    cv_strategy = 'standard'

# ============================
# 5. MODELS DEFINITION
# ============================
print("\n[5] Defining Models...")
models = {}

models['LogisticRegression'] = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, C=1.0))
])

models['RandomForest'] = RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_split=5,
    random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced'
)

if XGB_AVAILABLE:
    models['XGBoost'] = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1,
        eval_metric='mlogloss'
    )

# ============================
# 6. CROSS-VALIDATION TRAINING
# ============================
print("\n[6] Running Cross-Validation...")
results = {}
all_predictions = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    fold_accs, fold_f1s, fold_kappas = [], [], []
    all_fold_preds, all_fold_true = [], []

    splits = cv.split(X_fe, y, groups) if cv_strategy == 'group' else cv.split(X_fe, y)
    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        X_train, X_val = X_fe.iloc[train_idx], X_fe.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average='weighted')
        kappa = cohen_kappa_score(y_val, preds)

        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_kappas.append(kappa)
        all_fold_preds.extend(preds)
        all_fold_true.extend(y_val.values)

    results[name] = {
        'accuracy': np.mean(fold_accs),
        'f1': np.mean(fold_f1s),
        'kappa': np.mean(fold_kappas),
        'accuracy_std': np.std(fold_accs)
    }
    all_predictions[name] = (np.array(all_fold_true), np.array(all_fold_preds))
    print(f"  -> {name}: {results[name]['accuracy']*100:.2f}% (+/- {results[name]['accuracy_std']*100:.2f}%)")

# ============================
# 7. RESULTS SUMMARY
# ============================
print("\n" + "=" * 70)
print("FINAL CV RESULTS")
print("=" * 70)
for name, res in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
    print(f"{name:<20} Accuracy: {res['accuracy']*100:.2f}% | F1: {res['f1']*100:.2f}% | Kappa: {res['kappa']:.4f}")

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
print(f"\n🏆 Best model by CV: {best_model_name} ({results[best_model_name]['accuracy']*100:.2f}%)")

# ============================
# 8. SAVING ARTIFACTS FOR FASTAPI
# ============================
print("\n[8] Training best model on FULL dataset & Exporting artifacts...")
os.makedirs('models', exist_ok=True)

final_scaler = StandardScaler()
X_fe_scaled = final_scaler.fit_transform(X_fe)

# Выбираем и обучим финальный экземпляр лучшей модели
if best_model_name == 'XGBoost':
    final_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1
    )
    final_model.fit(X_fe, y)
elif best_model_name == 'RandomForest':
    final_model = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=5,
        random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced'
    )
    final_model.fit(X_fe, y)
else:
    final_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    final_model.fit(X_fe_scaled, y)

# Сохраняем модель и скалер
joblib.dump(final_model, 'models/model.pkl')
joblib.dump(final_scaler, 'models/scaler.pkl')

# Сохраняем метаданные (список колонок и названия классов)
meta_data = {
    'best_model': best_model_name,
    'cv_accuracy': round(float(results[best_model_name]['accuracy']), 4),
    'feature_names': list(X_fe.columns),
    'target_names': TARGET_NAMES
}

with open('models/model_meta.json', 'w', encoding='utf-8') as f:
    json.dump(meta_data, f, ensure_ascii=False, indent=2)

print("  Saved: models/model.pkl")
print("  Saved: models/scaler.pkl")
print("  Saved: models/model_meta.json")
print("\n✅ All artifacts created successfully! API is now ready to use real ML predictions.")