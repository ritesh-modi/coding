import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


data = {
    'feature1': [2.5, 3.1, 1.8, 4.2, 2.9, 3.5, 1.5, 4.0, 2.2, 3.8],
    'feature2': [10, 15, 8, 20, 12, 18, 7, 22, 9, 19],
    'feature3': [0.5, 0.8, 0.3, 0.9, 0.6, 0.85, 0.25, 0.95, 0.4, 0.88],
    'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # Binary classification
}

df = pd.DataFrame(data)

X = df[["feature1", "feature2", "feature3"]]
Y= df["target"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)


rf = RandomForestClassifier(random_state=42)

cv_scores = cross_validate(rf, scaled_X_train, Y_train, cv=3, return_train_score=True)
print(cv_scores["test_score"])
print(cv_scores)
