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


Feature selection - feature enginering
grid search - hyperparameter tuning
"""
import pandas as pd
import os
import joblib
# Set backend for Keras 3
os.environ["KERAS_BACKEND"] = "tensorflow"

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# TensorFlow / Keras wrapper
from keras_classifier import KerasBinaryClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# Ignore warnings for cleaner output
import warnings
# 1. Ignore specific Scikit-Learn FutureWarnings about version 1.7/1.8
warnings.filterwarnings("ignore", category=FutureWarning)
# 2. Ignore DeprecationWarnings (like the ones from Sklearn and Keras)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# 3. Ignore UserWarnings (like the Keras input_shape warning)
warnings.filterwarnings("ignore", category=UserWarning)
# 4. Specifically for the OpenSSL/urllib3 warning you see on macOS
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

# ============================================
# CONFIGURATION (will probably be in env)
# ============================================
USE_OVERSAMPLING = True
REMOVE_OUTLIERS = False
PRIMARY_SCORING = "f1"
# Options:
# "recall"            -> minimize false negatives (medical screening)
# "precision"         -> minimize false positives
# "f1"                -> balanced UX (recommended for website)
# "accuracy"          -> naive baseline

# ============================================
# 1. LOAD DATA
# ============================================

df = pd.read_csv("../Datasets/merged_cancer.csv")

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

symptom_cols = [
    "smoking", "yellow_fingers", "anxiety", "peer_pressure",
    "chronic_disease", "fatigue", "allergy", "wheezing",
    "alcohol", "coughing", "shortness_of_breath",
    "swallowing_difficulty", "chest_pain"
]

# Total symptom burden
df["symptom_count"] = df[symptom_cols].sum(axis=1)

# Lung / respiratory focused symptoms
df["respiratory_score"] = (
    df["coughing"] +
    df["wheezing"] +
    df["shortness_of_breath"] +
    df["chest_pain"]
)

# Age-based risk flag
df["age_risk"] = (df["age"] >= 60).astype(int)

# ============================================
# REMOVE OUTLIERS (optional - does not rly immprove things)
# ============================================
if REMOVE_OUTLIERS:
    df = df[df['age'].between(29, 80)]

# ============================================
# HELPER FUNCTIONS
# ============================================
def make_pipeline(model, use_oversampling):
    steps = [("preprocess", preprocessor)]

    if use_oversampling:
        steps.append(("oversample", RandomOverSampler(random_state=42)))

    steps.append(("model", model))

    # Use imblearn pipeline ONLY if oversampling is present
    return ImbPipeline(steps) if use_oversampling else Pipeline(steps)


# ============================================
# SPLIT FEATURES / TARGET
# ============================================

X = df.drop("lung_cancer", axis=1)
y = df["lung_cancer"]

numeric_features = ["age", "symptom_count", "respiratory_score"]
binary_features = [col for col in X.columns if col not in numeric_features]
preprocessor = ColumnTransformer(
    transformers=[
        ("age_scaler", StandardScaler(), numeric_features),
        ("binary", "passthrough", binary_features)
    ]
)

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
X_train_eval, y_train_eval = X_train, y_train

# ============================================
# 3. CUSTOM TENSORFLOW WRAPPER
# ============================================
# This class makes Keras behave like a standard Sklearn model

# ============================================
# 4. BUILD A PIPELINE (CORRECT WAY!)
#    Scaling happens INSIDE CV for each fold.
# ============================================

models = {
    "TensorFlow (Keras)": make_pipeline(
        KerasBinaryClassifier(epochs=50, batch_size=16, verbose=0),
        USE_OVERSAMPLING
    ),
    "KNN": make_pipeline(
        KNeighborsClassifier(n_neighbors=5),
        USE_OVERSAMPLING
    ),
    "Logistic Regression": make_pipeline(
        LogisticRegression(max_iter=1000),
        USE_OVERSAMPLING
    ),
    "SVM (RBF)": make_pipeline(
        SVC(kernel="rbf", probability=True),
        USE_OVERSAMPLING
    ),
    "Neural Network (MLP)": make_pipeline(
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=1000,
            random_state=42
        ),
        USE_OVERSAMPLING
    ),

    # -----------------------------
    # Tree-based models (NO scaling needed)
    # -----------------------------
    "Random Forest": (
        RandomForestClassifier(n_estimators=200, random_state=42)
        if not USE_OVERSAMPLING
        else ImbPipeline([
            ("preprocess", preprocessor),
            ("oversample", RandomOverSampler(random_state=42)),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42))
        ])
    ),
    "AdaBoost": (
        AdaBoostClassifier(n_estimators=200, random_state=42)
        if not USE_OVERSAMPLING
        else ImbPipeline([
            ("preprocess", preprocessor),
            ("oversample", RandomOverSampler(random_state=42)),
            ("model", AdaBoostClassifier(n_estimators=200, random_state=42))
        ])
    ),
    "Bernoulli Naive Bayes": (
        BernoulliNB()
        if not USE_OVERSAMPLING
        else ImbPipeline([
            ("preprocess", preprocessor),
            ("oversample", RandomOverSampler(random_state=42)),
            ("model", BernoulliNB())
        ])
    ),

    # -----------------------------
    # XGBoost (usually better WITHOUT ROS)
    # -----------------------------
    "XGBoost": make_pipeline(
        XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        ),
        False # XGBoost handles imbalance internally
    )
}

# ============================================
# 5. EVALUATING AND COMPARING MODELS
# ============================================

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision"
}
results = []
print(f"Training {len(models)} models...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"Running {name}...")
    cv_results = cross_validate(
        model,
        X_train_eval,
        y_train_eval,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    # Fit on full training set
    model.fit(X_train_eval, y_train_eval)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # y_pred = model.predict(X_test)
    # y_pred = (y_proba >= 0.35).astype(int)
    y_pred = (y_proba >= 0.5).astype(int)

    # Dump results
    save_path = f"Models/{name}.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)

    results.append({
        "Model": name,

        "CV Accuracy": cv_results["test_accuracy"].mean(),
        "CV Precision": cv_results["test_precision"].mean(),
        "CV Recall": cv_results["test_recall"].mean(),
        "CV F1": cv_results["test_f1"].mean(),
        "CV ROC-AUC": cv_results["test_roc_auc"].mean(),
        "CV PR-AUC": cv_results["test_pr_auc"].mean(),

        "Test Accuracy": accuracy_score(y_test, y_pred),
        "Test Precision": precision_score(y_test, y_pred),
        "Test Recall": recall_score(y_test, y_pred),
        "Test F1": f1_score(y_test, y_pred),
        "Test ROC-AUC": roc_auc_score(y_test, y_proba),
        "Test PR-AUC": average_precision_score(y_test, y_proba),
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