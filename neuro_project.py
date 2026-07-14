import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

sns.set_theme(style="whitegrid")

print("Loading dataset...")
try:
    df = pd.read_csv('mental-state.csv')
except FileNotFoundError:
    print("Error: 'mental-state.csv' not found in current directory.")
    exit()

print(f"Dataset shape: {df.shape[0]} samples, {df.shape[1] - 1} features.")

target_mapping = {0.0: 'Relax/Neutral', 1.0: 'Concentration', 2.0: 'Mental Fatigue'}
df['State_Name'] = df['Label'].map(target_mapping)

X = df.drop(columns=['Label', 'State_Name'])
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining models...")
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

if XGB_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, eval_metric='mlogloss'
    )

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    results[name] = acc
    print(f"  {name} - Accuracy: {acc:.2f}%")

best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]

print("-" * 40)
print(f"Best model: {best_model_name} ({best_accuracy:.2f}%)")
print("-" * 40)

print("\nFeature importance analysis...")
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("Top 5 features:")
for i in range(5):
    col_name = X.columns[indices[i]]
    val = importances[indices[i]] * 100
    print(f"  {i+1}. {col_name}: {val:.2f}%")

print("\nSaving plots...")
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['#34495e', '#2ecc71', '#3498db']
    bars = axes[0].bar(
        list(results.keys()), list(results.values()), 
        color=colors[:len(results)], width=0.4, edgecolor='black'
    )
    axes[0].set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_ylim(0, 110)
    for bar in bars:
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width()/2.0, height + 1, 
            f'{height:.2f}%', ha='center', va='bottom', fontsize=9
        )

    top_k = 10
    top_indices = indices[:top_k][::-1]
    axes[1].barh(range(top_k), importances[top_indices], color='#9b59b6', edgecolor='black', height=0.6)
    axes[1].set_yticks(range(top_k))
    axes[1].set_yticklabels([X.columns[i] for i in top_indices], fontsize=9)
    axes[1].set_xlabel('Importance score')
    axes[1].set_title('Top 10 Feature Importances', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('itmo_neuro_report.png', dpi=300)
    print("Saved 'itmo_neuro_report.png'")

    plt.figure(figsize=(7, 5))
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=list(target_mapping.values()),
        yticklabels=list(target_mapping.values()),
        annot_kws={"size": 11}
    )
    plt.title(f'Confusion Matrix ({best_model_name})', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix_itmo.png', dpi=300)
    print("Saved 'confusion_matrix_itmo.png'")

except Exception as e:
    print(f"Error during visualization: {e}")

print("\nProcess finished.")