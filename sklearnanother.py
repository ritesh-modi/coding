"""
CORRECTED sklearn example with proper train_test_split
This demonstrates the RIGHT way to:
- Split data into train and test sets
- Use GridSearchCV on training data only
- Evaluate on held-out test data
- Avoid data leakage
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CREATE SAMPLE DATA (Larger dataset for meaningful split)
# ============================================================================
print("=" * 70)
print("STEP 1: CREATE SAMPLE DATA")
print("=" * 70)

# Create 100 samples instead of 10 for meaningful train/test split
np.random.seed(42)

n_samples = 100
feature1 = np.random.uniform(1, 5, n_samples)
feature2 = np.random.uniform(5, 25, n_samples)
feature3 = np.random.uniform(0, 1, n_samples)

# Create target based on features (with some logic)
target = ((feature1 > 3) & (feature3 > 0.5)).astype(int)

data = {
    'feature1': feature1,
    'feature2': feature2,
    'feature3': feature3,
    'target': target
}

df = pd.DataFrame(data)
print("\nDataset (first 10 rows):")
print(df.head(10))
print(f"\nDataset shape: {df.shape}")
print(f"Target distribution:\n{df['target'].value_counts()}")

# ============================================================================
# 2. TRAIN-TEST SPLIT (CRITICAL STEP!)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: TRAIN-TEST SPLIT")
print("=" * 70)

# Separate features and target
X = df[['feature1', 'feature2', 'feature3']].values
y = df['target'].values

# Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,
    stratify=y          # Keep same class distribution
)

print(f"Original data shape: {X.shape}")
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"\nTraining target distribution:\n{pd.Series(y_train).value_counts()}")
print(f"\nTest target distribution:\n{pd.Series(y_test).value_counts()}")

# ============================================================================
# 3. MANUAL PREPROCESSING (FIT ON TRAIN, TRANSFORM BOTH)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: MANUAL DATA PREPROCESSING")
print("=" * 70)

# IMPORTANT: Fit scaler ONLY on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit + transform on train

# Transform test data using parameters learned from training data
X_test_scaled = scaler.transform(X_test)  # Only transform (no fit!)

print(f"\nScaler learned from training data:")
print(f"Mean: {scaler.mean_}")
print(f"Std: {scaler.scale_}")

print("\nTraining data (first 3 samples, scaled):")
print(X_train_scaled[:3])
print("\nTest data (first 3 samples, scaled):")
print(X_test_scaled[:3])

# ============================================================================
# 4. CROSS-VALIDATION ON TRAINING DATA ONLY
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: CROSS-VALIDATION (Training Data Only)")
print("=" * 70)

# Cross-validation on TRAINING data only
rf_model = RandomForestClassifier(random_state=42, n_estimators=50)

cv_scores = cross_val_score(
    rf_model, 
    X_train_scaled,  # Only training data!
    y_train,         # Only training labels!
    cv=5,
    scoring='accuracy'
)

print("\nRandom Forest - 5-Fold CV Results (on training data):")
print(f"Scores per fold: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# ============================================================================
# 5. GRID SEARCH CV ON TRAINING DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: GRID SEARCH CV (Training Data Only)")
print("=" * 70)

param_grid_rf = {
    'n_estimators': [30, 50, 100],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5]
}

print("\nParameter Grid:")
print(param_grid_rf)

# GridSearchCV on TRAINING data only
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_rf,
    cv=5,  # 5-fold CV within training data
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# Fit on training data only
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score (on training data): {grid_search.best_score_:.3f}")

# Get best model
best_rf_model = grid_search.best_estimator_

# ============================================================================
# 6. TRAIN SECOND MODEL (GRADIENT BOOSTING)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: GRADIENT BOOSTING - GRID SEARCH CV")
print("=" * 70)

param_grid_gb = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

grid_search_gb = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid_gb,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

grid_search_gb.fit(X_train_scaled, y_train)

print(f"\nBest Parameters: {grid_search_gb.best_params_}")
print(f"Best CV Score (on training data): {grid_search_gb.best_score_:.3f}")

best_gb_model = grid_search_gb.best_estimator_

# ============================================================================
# 7. FINAL EVALUATION ON TEST DATA (UNSEEN!)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: FINAL EVALUATION ON TEST DATA")
print("=" * 70)

# Predictions on TEST data (never seen during training/tuning)
y_pred_rf = best_rf_model.predict(X_test_scaled)
y_pred_gb = best_gb_model.predict(X_test_scaled)

print("\n" + "="*50)
print("RANDOM FOREST - TEST SET PERFORMANCE")
print("="*50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_rf):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_rf, zero_division=0):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred_rf, zero_division=0):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_rf, zero_division=0):.3f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\n" + "="*50)
print("GRADIENT BOOSTING - TEST SET PERFORMANCE")
print("="*50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_gb):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_gb, zero_division=0):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred_gb, zero_division=0):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_gb, zero_division=0):.3f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_gb))

# ============================================================================
# 8. COMPARE TRAIN VS TEST PERFORMANCE (Check for Overfitting)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: TRAIN VS TEST COMPARISON (Overfitting Check)")
print("=" * 70)

# Get training predictions
y_train_pred_rf = best_rf_model.predict(X_train_scaled)
y_train_pred_gb = best_gb_model.predict(X_train_scaled)

comparison = pd.DataFrame({
    'Model': ['Random Forest', 'Random Forest', 'Gradient Boosting', 'Gradient Boosting'],
    'Dataset': ['Train', 'Test', 'Train', 'Test'],
    'Accuracy': [
        accuracy_score(y_train, y_train_pred_rf),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_train, y_train_pred_gb),
        accuracy_score(y_test, y_pred_gb)
    ],
    'F1 Score': [
        f1_score(y_train, y_train_pred_rf),
        f1_score(y_test, y_pred_rf),
        f1_score(y_train, y_train_pred_gb),
        f1_score(y_test, y_pred_gb)
    ]
})

print("\nModel Performance Comparison:")
print(comparison.to_string(index=False))

# Calculate overfitting gap
rf_gap = accuracy_score(y_train, y_train_pred_rf) - accuracy_score(y_test, y_pred_rf)
gb_gap = accuracy_score(y_train, y_train_pred_gb) - accuracy_score(y_test, y_pred_gb)

print(f"\nOverfitting Analysis:")
print(f"Random Forest gap (train - test): {rf_gap:.3f}")
print(f"Gradient Boosting gap (train - test): {gb_gap:.3f}")
print(f"\nNote: Large gap (>0.1) indicates overfitting")

# ============================================================================
# 9. KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. ✅ Used train_test_split to create separate train/test sets
2. ✅ Fitted scaler ONLY on training data
3. ✅ Used GridSearchCV on training data only
4. ✅ Evaluated final model on unseen test data
5. ✅ Compared train vs test performance to detect overfitting

This is the CORRECT workflow to avoid data leakage!
""")

print("=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)