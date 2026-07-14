"""
NEURO PROJECT v2.0 — ITMO STARS EDITION
EEG Mental State Classification with Deep Learning

Requirements:
    pip install pandas numpy matplotlib seaborn scikit-learn torch shap xgboost

Usage:
    python neuro_project_v2.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, cohen_kappa_score, roc_auc_score
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
DATA_PATH = 'mental-state.csv'  # <-- ЗАМЕНИ НА СВОЙ ПУТЬ
TARGET_COL = 'Label'
SUBJECT_COL = None  # Auto-detected: 'Subject_ID', 'subject', 'user', etc.
RANDOM_STATE = 42
N_SPLITS = 5
DEVICE = torch.device('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')

TARGET_NAMES = {0.0: 'Relax/Neutral', 1.0: 'Concentration', 2.0: 'Mental Fatigue'}

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
y = df[TARGET_COL].copy()
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

# Correlation heatmap (top 20 by variance)
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

# t-SNE (sample if large)
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
print("  Saved: output/01_eda_overview.png")

# Feature distributions by class (top 6)
top6 = X.var().nlargest(6).index
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for idx, feat in enumerate(top6):
    for cls in sorted(y.unique()):
        subset = X.loc[y == cls, feat]
        axes[idx].hist(subset, bins=30, alpha=0.5, label=f'Class {int(cls)}', edgecolor='black', linewidth=0.3)
    axes[idx].set_title(feat, fontweight='bold', fontsize=10)
    axes[idx].legend()
plt.suptitle('Feature Distributions by Class (Top 6 by Variance)', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('output/02_feature_distributions.png', bbox_inches='tight')
plt.close()
print("  Saved: output/02_feature_distributions.png")

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

# Engagement index
if band_cols['beta'] and band_cols['alpha'] and band_cols['theta']:
    X_fe['engagement_index'] = X_fe[band_cols['beta']].mean(axis=1) / (
        X_fe[band_cols['alpha']].mean(axis=1) + X_fe[band_cols['theta']].mean(axis=1) + 1e-8
    )

# Channel statistics
channels = list(set([c.split('_')[0] for c in feature_cols if '_' in c]))
if len(channels) > 5:
    for ch in channels[:10]:
        ch_cols = [c for c in feature_cols if c.startswith(ch + '_')]
        if len(ch_cols) >= 3:
            X_fe[f'{ch}_std'] = X_fe[ch_cols].std(axis=1)
            X_fe[f'{ch}_range'] = X_fe[ch_cols].max(axis=1) - X_fe[ch_cols].min(axis=1)

# Spectral entropy
X_fe['spectral_entropy'] = -(X_fe[feature_cols] / (X_fe[feature_cols].sum(axis=1).values.reshape(-1,1) + 1e-8)).apply(
    lambda x: -np.sum(x * np.log(x + 1e-8)), axis=1
)

print(f"  Features after engineering: {X_fe.shape[1]} (was {X.shape[1]})")

# ============================
# 4. CV SETUP (Subject-wise!)
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
# 5. EEG-2D-CNN SETUP
# ============================
print("\n[5] EEG-2D-CNN Architecture Setup...")

# Try to reshape features into (channels, bands) for CNN
def build_eeg_tensor(X_df, feature_cols):
    """Reshape tabular EEG features into (samples, 1, channels, bands) tensor."""
    # Extract channel and band names from columns like 'AF3_alpha', 'F4_beta'
    parsed = []
    for col in feature_cols:
        if '_' in col:
            parts = col.rsplit('_', 1)
            if len(parts) == 2 and parts[1].lower() in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                parsed.append((parts[0], parts[1].lower(), col))

    if not parsed:
        print("  Could not parse channel_band structure. EEG-CNN disabled.")
        return None, None, None

    channels = sorted(list(set([p[0] for p in parsed])))
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    # Build tensor
    n_samples = len(X_df)
    n_ch = len(channels)
    n_bands = len(bands)
    tensor = np.zeros((n_samples, 1, n_ch, n_bands))

    for i, ch in enumerate(channels):
        for j, band in enumerate(bands):
            col_name = f"{ch}_{band}"
            if col_name in X_df.columns:
                tensor[:, 0, i, j] = X_df[col_name].values

    print(f"  EEG tensor shape: {tensor.shape} (samples, 1, channels={n_ch}, bands={n_bands})")
    return tensor, channels, bands

EEG_TENSOR = None
EEG_CHANNELS = None
EEG_BANDS = None
if TORCH_AVAILABLE:
    EEG_TENSOR, EEG_CHANNELS, EEG_BANDS = build_eeg_tensor(X, feature_cols)

class EEG2DCNN(nn.Module):
    """CNN that treats EEG as 2D spatial-spectral images."""
    def __init__(self, n_channels, n_bands, n_classes=3, dropout=0.4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 1))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AdaptiveAvgPool2d((4, 3))

        self.flatten_dim = 64 * 4 * 3
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], num_classes=3, dropout=0.4):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ============================
# 6. MODELS DEFINITION
# ============================
print("\n[6] Defining Models...")

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

print(f"  Sklearn models: {list(models.keys())}")
if TORCH_AVAILABLE:
    print(f"  PyTorch models: MLP, EEG2DCNN")
else:
    print("  PyTorch not available — skipping DL models")

# ============================
# 7. TRAINING HELPERS
# ============================
def train_torch_model(model, X_train, y_train, X_val, y_val, epochs=120, batch_size=32, lr=0.001, patience=20):
    model = model.to(DEVICE)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    train_ds = TensorDataset(torch.FloatTensor(X_train_s), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val_s), torch.LongTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler

def eval_torch_model(model, X_test, y_test, scaler):
    model.eval()
    X_test_s = scaler.transform(X_test)
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X_test_s).to(DEVICE))
        preds = outputs.argmax(1).cpu().numpy()
    return preds

def train_eeg_cnn(model, X_train_tensor, y_train, X_val_tensor, y_val, epochs=120, batch_size=32, lr=0.001, patience=20):
    """Train EEG2DCNN with tensor inputs."""
    model = model.to(DEVICE)

    # Normalize per feature across training set
    train_mean = X_train_tensor.mean(axis=0, keepdims=True)
    train_std = X_train_tensor.std(axis=0, keepdims=True) + 1e-8
    X_train_norm = (X_train_tensor - train_mean) / train_std
    X_val_norm = (X_val_tensor - train_mean) / train_std

    train_ds = TensorDataset(torch.FloatTensor(X_train_norm), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val_norm), torch.LongTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_mean, train_std

def eval_eeg_cnn(model, X_test_tensor, y_test, train_mean, train_std):
    model.eval()
    X_test_norm = (X_test_tensor - train_mean) / train_std
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X_test_norm).to(DEVICE))
        preds = outputs.argmax(1).cpu().numpy()
    return preds

# ============================
# 8. CROSS-VALIDATION TRAINING
# ============================
print("\n[7] Running Cross-Validation...")

results = {}
all_predictions = {}

def run_cv_sklearn(name, model, X_data, y_data, groups_data):
    print(f"\n  Training {name}...")
    fold_accs, fold_f1s, fold_kappas = [], [], []
    all_fold_preds, all_fold_true = [], []

    fold_idx = 0
    splits = cv.split(X_data, y_data, groups_data) if cv_strategy == 'group' else cv.split(X_data, y_data)
    for train_idx, val_idx in splits:
        fold_idx += 1
        X_train, X_val = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_train, y_val = y_data.iloc[train_idx], y_data.iloc[val_idx]

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

        print(f"    Fold {fold_idx}: Acc={acc:.4f}, F1={f1:.4f}, Kappa={kappa:.4f}")

    results[name] = {
        'accuracy': np.mean(fold_accs),
        'f1': np.mean(fold_f1s),
        'kappa': np.mean(fold_kappas),
        'accuracy_std': np.std(fold_accs)
    }
    all_predictions[name] = (np.array(all_fold_true), np.array(all_fold_preds))
    print(f"  -> {name}: {results[name]['accuracy']*100:.2f}% (+/- {results[name]['accuracy_std']*100:.2f}%)")

# Run sklearn models
for name, model in models.items():
    run_cv_sklearn(name, model, X_fe, y, groups)

# Run MLP
if TORCH_AVAILABLE:
    print(f"\n  Training MLP (PyTorch)...")
    fold_accs, fold_f1s, fold_kappas = [], [], []
    all_fold_preds, all_fold_true = [], []
    fold_idx = 0
    splits = cv.split(X_fe, y, groups) if cv_strategy == 'group' else cv.split(X_fe, y)
    for train_idx, val_idx in splits:
        fold_idx += 1
        X_train, X_val = X_fe.iloc[train_idx].values, X_fe.iloc[val_idx].values
        y_train, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values

        model = MLPClassifier(input_dim=X_fe.shape[1], num_classes=len(np.unique(y)))
        model, scaler = train_torch_model(model, X_train, y_train, X_val, y_val, epochs=100, patience=15)
        preds = eval_torch_model(model, X_val, y_val, scaler)

        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average='weighted')
        kappa = cohen_kappa_score(y_val, preds)

        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_kappas.append(kappa)
        all_fold_preds.extend(preds)
        all_fold_true.extend(y_val)

        print(f"    Fold {fold_idx}: Acc={acc:.4f}, F1={f1:.4f}, Kappa={kappa:.4f}")

    results['MLP'] = {
        'accuracy': np.mean(fold_accs),
        'f1': np.mean(fold_f1s),
        'kappa': np.mean(fold_kappas),
        'accuracy_std': np.std(fold_accs)
    }
    all_predictions['MLP'] = (np.array(all_fold_true), np.array(all_fold_preds))
    print(f"  -> MLP: {results['MLP']['accuracy']*100:.2f}% (+/- {results['MLP']['accuracy_std']*100:.2f}%)")

# Run EEG-2D-CNN
if TORCH_AVAILABLE and EEG_TENSOR is not None:
    print(f"\n  Training EEG2DCNN (PyTorch)...")
    fold_accs, fold_f1s, fold_kappas = [], [], []
    all_fold_preds, all_fold_true = [], []
    fold_idx = 0
    splits = cv.split(X_fe, y, groups) if cv_strategy == 'group' else cv.split(X_fe, y)
    for train_idx, val_idx in splits:
        fold_idx += 1
        X_train_t, X_val_t = EEG_TENSOR[train_idx], EEG_TENSOR[val_idx]
        y_train, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values

        n_ch = len(EEG_CHANNELS)
        n_bands = len(EEG_BANDS)
        model = EEG2DCNN(n_channels=n_ch, n_bands=n_bands, n_classes=len(np.unique(y)))
        model, train_mean, train_std = train_eeg_cnn(model, X_train_t, y_train, X_val_t, y_val, epochs=100, patience=15)
        preds = eval_eeg_cnn(model, X_val_t, y_val, train_mean, train_std)

        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average='weighted')
        kappa = cohen_kappa_score(y_val, preds)

        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_kappas.append(kappa)
        all_fold_preds.extend(preds)
        all_fold_true.extend(y_val)

        print(f"    Fold {fold_idx}: Acc={acc:.4f}, F1={f1:.4f}, Kappa={kappa:.4f}")

    results['EEG2DCNN'] = {
        'accuracy': np.mean(fold_accs),
        'f1': np.mean(fold_f1s),
        'kappa': np.mean(fold_kappas),
        'accuracy_std': np.std(fold_accs)
    }
    all_predictions['EEG2DCNN'] = (np.array(all_fold_true), np.array(all_fold_preds))
    print(f"  -> EEG2DCNN: {results['EEG2DCNN']['accuracy']*100:.2f}% (+/- {results['EEG2DCNN']['accuracy_std']*100:.2f}%)")

# ============================
# 9. RESULTS TABLE
# ============================
print("\n[8] Model Comparison Results:")
print("-" * 65)
print(f"{'Model':<20} {'Accuracy':<12} {'F1-Score':<12} {'Cohen Kappa':<12}")
print("-" * 65)
for name, res in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
    print(f"{name:<20} {res['accuracy']*100:.2f}%       {res['f1']*100:.2f}%       {res['kappa']:.4f}")
print("-" * 65)

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
print(f"\n🏆 Best model: {best_model_name} ({results[best_model_name]['accuracy']*100:.2f}%)")

# ============================
# 10. VISUALIZATIONS
# ============================
print("\n[9] Saving comparison plots...")

n_models = len(results)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

# Accuracy comparison
ax = axes[0]
names = list(results.keys())
accs = [results[n]['accuracy']*100 for n in names]
stds = [results[n]['accuracy_std']*100 for n in names]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
bars = ax.bar(names, accs, yerr=stds, color=colors, edgecolor='black', capsize=4)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Accuracy Comparison (CV)', fontweight='bold')
ax.set_ylim(0, 110)
for bar, acc, std in zip(bars, accs, stds):
    ax.text(bar.get_x() + bar.get_width()/2, acc + std + 2, f'{acc:.1f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

# F1 vs Kappa
ax = axes[1]
f1s = [results[n]['f1']*100 for n in names]
kappas = [results[n]['kappa'] for n in names]
ax.scatter(kappas, f1s, s=200, c=colors, edgecolors='black', linewidth=2, zorder=5)
for i, name in enumerate(names):
    ax.annotate(name, (kappas[i], f1s[i]), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=8, fontweight='bold')
ax.set_xlabel('Cohen Kappa')
ax.set_ylabel('F1-Score (%)')
ax.set_title('F1 vs Kappa (Robustness)', fontweight='bold')
ax.grid(True, alpha=0.3)

# Confusion matrices for top models
for idx, model_name in enumerate(list(results.keys())[:4]):
    ax = axes[idx + 2]
    y_true, y_pred = all_predictions[model_name]
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=[f'C{i}' for i in range(len(np.unique(y)))],
                yticklabels=[f'C{i}' for i in range(len(np.unique(y)))],
                annot_kws={"size": 10})
    ax.set_title(f'{model_name}\nAcc: {results[model_name]["accuracy"]*100:.1f}%', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('output/03_model_comparison.png', bbox_inches='tight')
plt.close()
print("  Saved: output/03_model_comparison.png")

# Feature importance
print("\n[10] Feature Importance Analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

if 'RandomForest' in models:
    rf_model = models['RandomForest']
    rf_model.fit(X_fe, y)
    rf_imp = pd.Series(rf_model.feature_importances_, index=X_fe.columns).sort_values(ascending=False)
    top_rf = rf_imp.head(15)
    axes[0].barh(range(len(top_rf)), top_rf.values, color='#2ecc71', edgecolor='black')
    axes[0].set_yticks(range(len(top_rf)))
    axes[0].set_yticklabels(top_rf.index, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_title('Random Forest Feature Importance', fontweight='bold')
    axes[0].set_xlabel('Importance')

if XGB_AVAILABLE and 'XGBoost' in models:
    xgb_model = models['XGBoost']
    xgb_model.fit(X_fe, y)
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=X_fe.columns).sort_values(ascending=False)
    top_xgb = xgb_imp.head(15)
    axes[1].barh(range(len(top_xgb)), top_xgb.values, color='#3498db', edgecolor='black')
    axes[1].set_yticks(range(len(top_xgb)))
    axes[1].set_yticklabels(top_xgb.index, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_title('XGBoost Feature Importance', fontweight='bold')
    axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('output/04_feature_importance.png', bbox_inches='tight')
plt.close()
print("  Saved: output/04_feature_importance.png")

# SHAP
print("\n[11] SHAP Explainability...")
if SHAP_AVAILABLE and XGB_AVAILABLE and 'XGBoost' in models:
    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_fe.iloc[:200])

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_fe.iloc[:200], plot_type="bar", show=False)
        plt.title('SHAP Feature Importance (XGBoost)', fontweight='bold')
        plt.tight_layout()
        plt.savefig('output/05_shap_importance.png', bbox_inches='tight')
        plt.close()
        print("  Saved: output/05_shap_importance.png")

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_fe.iloc[:200], show=False)
        plt.title('SHAP Summary Plot (XGBoost)', fontweight='bold')
        plt.tight_layout()
        plt.savefig('output/06_shap_summary.png', bbox_inches='tight')
        plt.close()
        print("  Saved: output/06_shap_summary.png")
    except Exception as e:
        print(f"  SHAP error: {e}")
else:
    print("  SHAP/XGBoost not available — skipping")

# ============================
# 11. FINAL REPORT
# ============================
print("\n" + "=" * 70)
print("FINAL REPORT")
print("=" * 70)
print(f"Dataset: {df.shape[0]} samples, {X_fe.shape[1]} features, {y.nunique()} classes")
print(f"CV Strategy: {cv_strategy} ({N_SPLITS}-fold)")
print(f"Best Model: {best_model_name}")
print(f"Best Accuracy: {results[best_model_name]['accuracy']*100:.2f}%")
print(f"Best F1-Score: {results[best_model_name]['f1']*100:.2f}%")
print(f"Best Cohen Kappa: {results[best_model_name]['kappa']:.4f}")
print("\nGenerated files in output/:")
for f in ['01_eda_overview.png', '02_feature_distributions.png', '03_model_comparison.png',
          '04_feature_importance.png', '05_shap_importance.png', '06_shap_summary.png']:
    print(f"  {f}")
print("=" * 70)
print("\n✅ Done! Next steps:")
print("   1. Streamlit dashboard (neuro_dashboard.py)")
print("   2. FastAPI inference API")
print("   3. GitHub repo + README")
print("=" * 70)