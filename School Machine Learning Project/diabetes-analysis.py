"""
Task 2: Diabetes Classification - Step-by-Step Analysis
Run each section individually to follow the lab requirements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import os
from datetime import datetime

# Create results directory
os.makedirs('results', exist_ok=True)

# Set style and seed
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)

print("\n" + "="*80)
print("DIABETES CLASSIFICATION ANALYSIS - STEP BY STEP")
print("="*80 + "\n")

# print("\n" + "="*80)
print("SECTION 1: DATASET DESCRIPTION & STATISTICAL SUMMARY")
print("="*80 + "\n")

# Load dataset
try:
    df = pd.read_csv('diabetes.csv')
    print("[OK] Dataset loaded successfully\n")
except FileNotFoundError:
    print("[ERROR] File 'data/diabetes.csv' not found!")
    print("Please ensure diabetes.csv is in the 'data/' folder")
    exit()

# 1.1 Basic Information
print("--- 1.1 Dataset Overview ---")
print(f"Number of rows: {df.shape}")
print(f"Number of columns: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
# 1.3 Statistical Summary
print("\n--- 1.3 Statistical Summary ---")
print(df.describe())
# 1.4 Target Distribution
print("\n--- 1.4 Target Variable Distribution ---")
print(f"\nOutcome counts:")
print(df['Outcome'].value_counts())
print(f"\nOutcome percentages:")
print(df['Outcome'].value_counts(normalize=True) * 100)

print("\n" + "="*80)
print("SECTION 2: OUTLIER DETECTION & HANDLING")
print("="*80 + "\n")

numeric_cols = df.columns[:-1]
# 2.2 IQR Method for Outlier Detection
print("\n--- 2.2 Outlier Count (IQR Method) ---")
outlier_summary = []

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_pct = (outlier_count / len(df)) * 100
    
    outlier_summary.append({
        'Feature': col,
        'Outlier_Count': outlier_count,
        'Outlier_Percentage': f"{outlier_pct:.2f}%"
    })
    
    print(f"{col}: {outlier_count} outliers ({outlier_pct:.2f}%)")

outlier_df = pd.DataFrame(outlier_summary)

# 3.1 Check for explicit NaN values
print("--- 3.1 Explicit Missing Values ---")
missing_explicit = df.isnull().sum()
print(missing_explicit)
print(f"\nTotal explicit missing values: {missing_explicit.sum()}")

# 3.2 Check for zeros (hidden missing values in medical data)
print("\n--- 3.2 Hidden Missing Values (Zeros in Medical Fields) ---")
medical_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

print("\nPhysiologically impossible zeros:")
for col in medical_cols:
    zero_count = (df[col] == 0).sum()
    zero_pct = (zero_count / len(df)) * 100
    print(f"{col}: {zero_count} zeros ({zero_pct:.2f}%)")

df_clean = df.copy()  # create a working copy before modifying

# Replace zeros with NaN
for col in medical_cols:
    df_clean[col] = df_clean[col].replace(0, np.nan)

print("\nMissing values after replacement:")
print(df_clean.isna().sum())

# Impute with median
imputer = SimpleImputer(strategy='median')
df_clean[medical_cols] = imputer.fit_transform(df_clean[medical_cols])

print("\nMissing values after imputation:")
print(df_clean.isna().sum())
print("\n[OK] All missing values handled successfully!")

# SECTION 4: CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: CORRELATION ANALYSIS")
print("="*80 + "\n")

# 4.1 Compute Correlation Matrix
print("--- 4.1 Correlation Matrix ---")
corr_matrix = df_clean.corr()
print(corr_matrix)
# 4.2 Correlation with Target
print("\n--- 4.2 Features Correlation with Outcome (Sorted) ---")
outcome_corr = corr_matrix['Outcome'].sort_values(ascending=False)
print(outcome_corr)

print("\n[INSIGHT] Top 3 positive correlations with Outcome:")
for i in range(1, 4):  # Skip Outcome itself
    print(f"  {i}. {outcome_corr.index[i]}: {outcome_corr.iloc[i]:.3f}")
# ============================================================================
# SECTION 5: NORMALIZATION & SCALING
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: NORMALIZATION & SCALING")
print("="*80 + "\n")

# 5.1 Why Scaling?
print("--- 5.1 Why Scaling is Necessary ---")
print("[REASON 1] Features have different scales (e.g., Age vs Insulin)")
print("[REASON 2] KNN and Naive Bayes are sensitive to feature scales")
print("[REASON 3] StandardScaler transforms to mean=0, std=1")

# 5.2 Before Scaling Statistics
print("\n--- 5.2 Before Scaling ---")
X = df_clean.drop('Outcome', axis=1)
y = df_clean['Outcome']

print("Feature means:")
print(X.mean())
print("\nFeature standard deviations:")
print(X.std())
# 5.3 Apply StandardScaler (will be done after train-test split)
print("\n--- 5.3 Scaling Strategy ---")
print("[STRATEGY] Apply StandardScaler AFTER train-test split")
print("[REASON] Prevent data leakage from test set to training set")
print("[METHOD] Fit scaler on training data, transform both train and test")

# ============================================================================
# SECTION 6: TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("SECTION 6: TRAIN-TEST SPLIT")
print("="*80 + "\n")

# 6.1 Split Strategy
print("--- 6.1 Split Configuration ---")
print("[SPLIT RATIO] 80% Training, 20% Testing")
print("[STRATIFY] Yes - preserve class distribution in both sets")
print("[RANDOM STATE] 42 (for reproducibility)")
# 6.2 Perform Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\n--- 6.2 Split Results ---")
train_n = X_train.shape[0]
test_n = X_test.shape[0]
total_n = len(X)

print(f"Training set: {train_n} samples ({train_n / total_n * 100:.1f}%)")
print(f"Test set: {test_n} samples ({test_n / total_n * 100:.1f}%)")

print("\nTraining set class distribution:")
print(y_train.value_counts())
print("\nTest set class distribution:")
print(y_test.value_counts())
# 6.3 Apply Scaling
print("\n--- 6.3 Apply StandardScaler ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("[OK] Scaler fitted on training data")
print("[OK] Training and test data scaled")

print("\nAfter scaling - Training set statistics:")
print(f"Mean (should be ~0): {X_train_scaled.mean(axis=0).round(3)}")
print(f"Std (should be ~1): {X_train_scaled.std(axis=0).round(3)}")

# ============================================================================
# SECTION 8: TRAIN ML MODELS
# ============================================================================
print("\n" + "="*80)
print("SECTION 8: TRAIN MACHINE LEARNING MODELS")
print("="*80 + "\n")

# 8.1 Train KNN
print("--- 8.1 Training KNN with Grid Search ---")
print("[INFO] Testing n_neighbors: [3, 5, 7, 9, 11]")
param_grid_knn = {'n_neighbors': [3, 5, 7, 9, 11]}
knn = KNeighborsClassifier(weights='distance')
grid_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='roc_auc', n_jobs=-1)
grid_knn.fit(X_train_scaled, y_train)

print(f"[RESULT] Best n_neighbors: {grid_knn.best_params_['n_neighbors']}")
print(f"[RESULT] Best CV AUC: {grid_knn.best_score_:.4f}")

knn_best = grid_knn.best_estimator_
y_pred_knn = knn_best.predict(X_test_scaled)
y_proba_knn = knn_best.predict_proba(X_test_scaled)[:, 1]
print("[OK] KNN trained successfully")
# 8.2 Train Naive Bayes
print("\n--- 8.2 Training Naive Bayes ---")
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)
y_proba_nb = nb.predict_proba(X_test_scaled)[:, 1]
print("[OK] Naive Bayes trained successfully")
# 8.3 Train Random Forest
print("\n--- 8.3 Training Random Forest ---")
print("[CONFIG] n_estimators=200, max_depth=None, min_samples_split=5")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    random_state=42,
    n_jobs=1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]
print("[OK] Random Forest trained successfully")

# 8.4 Feature Importance (Random Forest)
print("\n--- 8.4 Feature Importance (Random Forest) ---")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("Top 5 Most Important Features:")
print(feature_importance.head(5).to_string(index=False))
# ============================================================================
# SECTION 9: REPORT RESULTS & EVALUATION METRICS
# ============================================================================
print("\n" + "="*80)
print("SECTION 9: REPORT RESULTS & EVALUATION METRICS")
print("="*80 + "\n")

models = {
    'KNN': (y_pred_knn, y_proba_knn),
    'Naive Bayes': (y_pred_nb, y_proba_nb),
    'Random Forest': (y_pred_rf, y_proba_rf)
}

# 9.1 Classification Reports
print("--- 9.1 Classification Reports ---\n")
for model_name, (y_pred, y_proba) in models.items():
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")
    print(classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic']))

# 9.2 Confusion Matrices
print("\n--- 9.2 Confusion Matrices ---")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (model_name, (y_pred, _)) in enumerate(models.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Non-Diabetic', 'Diabetic'],
                yticklabels=['Non-Diabetic', 'Diabetic'])
    axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_xlabel('Predicted')

plt.suptitle("Confusion Matrices", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/09_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("[SAVED] Confusion matrices saved as 'results/09_confusion_matrices.png'")
plt.show()

# 9.3 ROC Curves
print("\n--- 9.3 ROC Curves ---")
plt.figure(figsize=(10, 7))

for model_name, (_, y_proba) in models.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2.5)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1.5)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/09_roc_curves.png', dpi=300, bbox_inches='tight')
print("[SAVED] ROC curves saved as 'results/09_roc_curves.png'")
plt.show()

# 9.4 Metrics Summary Table
print("\n--- 9.4 Comprehensive Metrics Summary ---")
metrics_list = []

for model_name, (y_pred, y_proba) in models.items():
    metrics_list.append({
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba)
    })

metrics_df = pd.DataFrame(metrics_list)
print("\n" + metrics_df.to_string(index=False))

# Save to CSV
metrics_df.to_csv('results/09_metrics_summary.csv', index=False)
print("\n[SAVED] Metrics saved as 'results/09_metrics_summary.csv'")

input("\n[PRESS ENTER TO CONTINUE TO SECTION 10...]")


# ============================================================================
# SECTION 10: RESULT IMPLICATION & RECOMMENDATION
# ============================================================================
print("\n" + "="*80)
print("SECTION 10: RESULT IMPLICATION & RECOMMENDATION")
print("="*80 + "\n")

# 10.1 Best Model Identification
print("--- 10.1 Best Model Identification ---")
best_model_idx = metrics_df['AUC-ROC'].idxmax()
best_model = metrics_df.loc[best_model_idx]

print(f"\n[BEST MODEL] {best_model['Model']}")
print(f"  - Accuracy:  {best_model['Accuracy']:.4f}")
print(f"  - Precision: {best_model['Precision']:.4f}")
print(f"  - Recall:    {best_model['Recall']:.4f}")
print(f"  - F1-Score:  {best_model['F1-Score']:.4f}")
print(f"  - AUC-ROC:   {best_model['AUC-ROC']:.4f}")

# 10.2 Clinical Context Discussion
print("\n--- 10.2 Clinical Context & Trade-offs ---")
print("\n[CONTEXT] Diabetes Screening Application")
print("  - False Negative (missed diabetic): DANGEROUS - patient doesn't get treatment")
print("  - False Positive (flagged as diabetic): LESS DANGEROUS - further testing clarifies")
print("  - Therefore: RECALL (sensitivity) is MORE important than Precision")

print("\n[RECALL COMPARISON]")
for _, row in metrics_df.iterrows():
    print(f"  {row['Model']}: Recall = {row['Recall']:.4f} "
          f"(catches {row['Recall']*100:.1f}% of diabetics)")

# 10.3 Model-Specific Insights
print("\n--- 10.3 Model-Specific Insights ---")
print("\n[KNN]")
print("  - Moderate performance")
print("  - Sensitive to local patterns in data")
print("  - Requires scaling (which we did)")

print("\n[Naive Bayes]")
print("  - Fast baseline model")
print("  - Assumes feature independence (may not hold in medical data)")
print("  - Generally lower performance but interpretable")

print("\n[Random Forest]")
print("  - Usually best performance")
print("  - Captures non-linear relationships")
print("  - Provides feature importance")

# 10.4 Final Recommendation
print("\n--- 10.4 Final Recommendation ---")
print(f"\n[RECOMMENDATION] Deploy {best_model['Model']} for diabetes screening")
print("\n[REASONS]")
print(f"  1. Highest AUC-ROC ({best_model['AUC-ROC']:.4f}) - best overall discrimination")
print(f"  2. Strong recall ({best_model['Recall']:.4f}) - minimizes missed diabetics")
print(f"  3. Balanced precision ({best_model['Precision']:.4f}) - reduces false alarms")

print("\n[DEPLOYMENT CONSIDERATIONS]")
print("  1. Re-train model periodically with new patient data")
print("  2. Monitor false negative rate in production")
print("  3. Use probability threshold tuning for recall/precision trade-off")
print("  4. Combine with clinical judgment for final diagnosis")

# 10.5 Summary Statistics
print("\n--- 10.5 Analysis Summary ---")
print(f"\nDataset: {len(df)} patients")
print(f"Features: {X.shape} clinical measurements")
print(f"Models trained: {len(models)}")
print(f"Best model: {best_model['Model']}")
print(f"Best AUC: {best_model['AUC-ROC']:.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nAll results saved in 'results/' folder:")
print("  - 01_class_distribution.png")
print("  - 02_outlier_boxplots.png")
print("  - 03_distributions_no_zeros.png")
print("  - 04_correlation_heatmap.png")
print("  - 09_confusion_matrices.png")
print("  - 09_roc_curves.png")
print("  - 09_metrics_summary.csv")

print("\n[OK] Step-by-step analysis complete!")
print("You can now write your report using these outputs.\n")