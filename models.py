"""
Train and compare models properly using best ML practices.
This version avoids data leakage, randomness issues, and scaling mistakes.

lil present: https://www.youtube.com/watch?v=KUZ7jG7BKE8&list=RDKUZ7jG7BKE8&start_radio=1

Models to implement: 
- K-Nearest Neighbors (KNN)
- Bernoulli Naive Bayes
- Some tree-based model (Decision Tree, Random Forest, XGBoost, etc.)
- Logistic Regression
- Support Vector Machine (SVM)
- Neural Network (MLP)

Main focus should be the "recall" metric due to our requirement for accuracy lung cancer detection.
"""

import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================
# CONFIGURATION (will probably be in env)
# ============================================
USE_OVERSAMPLING = False
REMOVE_OUTLIERS = False
PRIMARY_SCORING = "recall"
# Options:
# "recall"            -> minimize false negatives (medical screening)
# "precision"         -> minimize false positives
# "f1"                -> balanced UX (recommended for website)
# "accuracy"          -> naive baseline

# ============================================
# 1. LOAD DATA
# ============================================

df = pd.read_csv("Datasets/merged_cancer.csv")

X = df.drop("lung_cancer", axis=1)
y = df["lung_cancer"]

# ============================================
# REMOVE OUTLIERS (optional - does not rly immprove things)
# ============================================
if REMOVE_OUTLIERS:
    age_min = 29
    age_max = 80
    df = df[df['age'].between(age_min, age_max)]

# ============================================
# 2. SPLIT BEFORE OVERSAMPLING
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================
# 3. OVERSAMPLE *ONLY THE TRAINING SET*
# ============================================

if USE_OVERSAMPLING:
    ros = RandomOverSampler(random_state=42)
    X_train_eval, y_train_eval = ros.fit_resample(X_train, y_train)
else:
    X_train_eval, y_train_eval = X_train, y_train

# ============================================
# 4. BUILD A PIPELINE (CORRECT WAY!)
#    Scaling happens INSIDE CV for each fold.
# ============================================

models = {
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,   # oversampling already handled
        eval_metric="logloss",
        random_state=42
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Bernoulli Naive Bayes": BernoulliNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "AdaBoost": AdaBoostClassifier(
        n_estimators=200,
        random_state=42
    ),

    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf"))
    ]),

    "Neural Network (MLP)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42
        ))
    ])
}

# ============================================
# 5. EVALUATING AND COMPARING MODELS
# ============================================

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}
results = []

for name, model in models.items():
    cv_results = cross_validate(
        model,
        X_train_eval,
        y_train_eval,
        cv=5,
        scoring=scoring,
        return_train_score=False
    )

    # Fit on full training set
    model.fit(X_train_eval, y_train_eval)

    y_pred = model.predict(X_test)

    results.append({
        "Model": name,

        # Cross-validated metrics
        "CV Accuracy": cv_results["test_accuracy"].mean(),
        "CV Precision": cv_results["test_precision"].mean(),
        "CV Recall": cv_results["test_recall"].mean(),
        "CV F1": cv_results["test_f1"].mean(),

        # Test metrics
        "Test Accuracy": accuracy_score(y_test, y_pred),
        "Test Precision": precision_score(y_test, y_pred),
        "Test Recall": recall_score(y_test, y_pred),
        "Test F1": f1_score(y_test, y_pred),
    })

results_df = pd.DataFrame(results)

sort_column = {
    "accuracy": "Test Accuracy",
    "precision": "Test Precision",
    "recall": "Test Recall",
    "f1": "Test F1"
}[PRIMARY_SCORING]

results_df = results_df.sort_values(by=sort_column, ascending=False)

print(results_df)