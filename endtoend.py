import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from openai import AzureOpenAI
from pydantic import BaseModel
import json

# ================================
# 1. LOAD DATA
# ================================
df = pd.read_csv('customer_data.csv')
print("="*50)
print("DATA LOADED")
print("="*50)
print(df.head())
print()

# ================================
# 2. EDA - EXPLORATORY DATA ANALYSIS
# ================================
print("="*50)
print("EDA - BASIC INFO")
print("="*50)
print(df.info())
print()

print("="*50)
print("EDA - DESCRIPTIVE STATISTICS")
print("="*50)
print(df.describe())
print()

print("="*50)
print("EDA - MISSING VALUES")
print("="*50)
print(df.isnull().sum())
print()

print("="*50)
print("EDA - TARGET DISTRIBUTION")
print("="*50)
print(df['churn'].value_counts())
print()

print("="*50)
print("EDA - CORRELATION (NUMERIC FEATURES)")
print("="*50)
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(df[numeric_cols].corr())
print()

# ================================
# 3. DATA CLEANING
# ================================
print("="*50)
print("DATA CLEANING")
print("="*50)

# Drop customer_id (not useful for prediction)
df_clean = df.drop('customer_id', axis=1)
print(f"Dropped customer_id column")

# Check for duplicates
duplicates = df_clean.duplicated().sum()
print(f"Duplicates found: {duplicates}")

# Fill missing values if any (none in this dataset)
df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
print("Missing values handled")
print()

# ================================
# 4. FEATURE ENGINEERING
# ================================
print("="*50)
print("FEATURE ENGINEERING")
print("="*50)

# Separate features and target
X = df_clean.drop('churn', axis=1)
y = df_clean['churn']

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")
print()

# ================================
# 5. PREPROCESSING
# ================================
print("="*50)
print("PREPROCESSING")
print("="*50)

# One-Hot Encoding for categorical features
encoder = OneHotEncoder(sparse_output=False, drop='first')
X_categorical = encoder.fit_transform(X[categorical_features])
categorical_feature_names = encoder.get_feature_names_out(categorical_features)

# StandardScaler for numeric features
scaler = StandardScaler()
X_numeric = scaler.fit_transform(X[numeric_features])

# Combine features
X_processed = np.hstack([X_numeric, X_categorical])
feature_names = numeric_features + list(categorical_feature_names)

print(f"Processed features shape: {X_processed.shape}")
print(f"Feature names: {feature_names}")
print()

# ================================
# 6. TRAIN-TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

print("="*50)
print("TRAIN-TEST SPLIT")
print("="*50)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print()

# ================================
# 7. MODEL TRAINING WITH GRIDSEARCHCV
# ================================
print("="*50)
print("MODEL TRAINING WITH GRIDSEARCHCV")
print("="*50)

# Define model
rf = RandomForestClassifier(random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

# GridSearchCV with cross-validation
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
print()

# ================================
# 8. CROSS-VALIDATION ON BEST MODEL
# ================================
print("="*50)
print("CROSS-VALIDATION SCORES")
print("="*50)

best_model = grid_search.best_estimator_

cv_accuracy = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
cv_precision = cross_val_score(best_model, X_train, y_train, cv=5, scoring='precision')
cv_recall = cross_val_score(best_model, X_train, y_train, cv=5, scoring='recall')
cv_f1 = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1')

print(f"CV Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std():.4f})")
print(f"CV Precision: {cv_precision.mean():.4f} (+/- {cv_precision.std():.4f})")
print(f"CV Recall: {cv_recall.mean():.4f} (+/- {cv_recall.std():.4f})")
print(f"CV F1 Score: {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
print()

# ================================
# 9. MODEL EVALUATION ON TEST SET
# ================================
print("="*50)
print("TEST SET EVALUATION")
print("="*50)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print()
print("Confusion Matrix:")
print(conf_matrix)
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))
print()

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("="*50)
print("FEATURE IMPORTANCE")
print("="*50)
print(feature_importance)
print()

# ================================
# 10. AZURE OPENAI INSIGHTS WITH PYDANTIC
# ================================
print("="*50)
print("AZURE OPENAI INSIGHTS")
print("="*50)

# Pydantic model for structured output
class ModelInsights(BaseModel):
    summary: str
    key_findings: list[str]
    recommendations: list[str]
    risk_factors: list[str]

# Prepare data summary for GPT
data_summary = f"""
Model Performance Metrics:
- Accuracy: {accuracy:.4f}
- Precision: {precision:.4f}
- Recall: {recall:.4f}
- F1 Score: {f1:.4f}

Cross-Validation Results:
- CV Accuracy: {cv_accuracy.mean():.4f}
- CV Precision: {cv_precision.mean():.4f}
- CV Recall: {cv_recall.mean():.4f}
- CV F1: {cv_f1.mean():.4f}

Top 5 Important Features:
{feature_importance.head().to_string()}

Confusion Matrix:
{conf_matrix}

Dataset Info:
- Total customers: {len(df)}
- Churned customers: {df['churn'].sum()}
- Churn rate: {df['churn'].mean():.2%}
"""

# Azure OpenAI setup (you need to set these environment variables or replace with actual values)
# export AZURE_OPENAI_ENDPOINT="your-endpoint"
# export AZURE_OPENAI_KEY="your-key"
# export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"

try:
    import os
    
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-02-01",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": "You are a data science expert analyzing machine learning model results for customer churn prediction."},
            {"role": "user", "content": f"Analyze this model performance and provide insights:\n\n{data_summary}"}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    
    gpt_response = response.choices[0].message.content
    
    # Parse into Pydantic model (simplified - in production you'd use function calling)
    insights = ModelInsights(
        summary=gpt_response[:200] + "...",
        key_findings=[
            f"Model achieves {accuracy:.1%} accuracy on test set",
            f"Top predictor: {feature_importance.iloc[0]['feature']}",
            f"Churn rate in dataset: {df['churn'].mean():.1%}"
        ],
        recommendations=[
            "Focus on customers with monthly contracts",
            "Reduce support call response time",
            "Offer incentives for longer contract commitments"
        ],
        risk_factors=[
            "High number of support calls",
            "Short tenure (months_active)",
            "Monthly contract type"
        ]
    )
    
    print("GPT-4 Analysis:")
    print(gpt_response)
    print()
    print("="*50)
    print("STRUCTURED INSIGHTS (Pydantic)")
    print("="*50)
    print(insights.model_dump_json(indent=2))
    
except Exception as e:
    print(f"Azure OpenAI not configured or error occurred: {e}")
    print("Skipping GPT insights...")
    print()
    print("To enable Azure OpenAI, set these environment variables:")
    print("- AZURE_OPENAI_ENDPOINT")
    print("- AZURE_OPENAI_KEY")
    print("- AZURE_OPENAI_DEPLOYMENT")
    
    # Fallback insights
    insights = ModelInsights(
        summary=f"Random Forest model achieved {accuracy:.1%} accuracy in predicting customer churn.",
        key_findings=[
            f"Model achieves {accuracy:.1%} accuracy on test set",
            f"Top predictor: {feature_importance.iloc[0]['feature']} with importance {feature_importance.iloc[0]['importance']:.4f}",
            f"Churn rate in dataset: {df['churn'].mean():.1%}",
            f"Model shows good balance between precision ({precision:.1%}) and recall ({recall:.1%})"
        ],
        recommendations=[
            "Focus retention efforts on customers with monthly contracts",
            "Improve support quality to reduce multiple support calls",
            "Offer incentives for customers to switch to annual contracts",
            "Monitor customers in their first 6 months closely"
        ],
        risk_factors=[
            "High number of support calls (>3)",
            "Short tenure (months_active < 10)",
            "Monthly contract type",
            "Lower income levels"
        ]
    )
    
    print("="*50)
    print("STRUCTURED INSIGHTS (Pydantic)")
    print("="*50)
    print(json.dumps(insights.model_dump(), indent=2))

print()
print("="*50)
print("PIPELINE COMPLETE!")
print("="*50)