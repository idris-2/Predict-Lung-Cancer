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
from sklearn.model_selection import train_test_split, cross_val_score
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
from sklearn.metrics import accuracy_score, recall_score

# ============================================
# 1. LOAD DATA
# ============================================

df = pd.read_csv("Datasets/merged_cancer.csv")

X = df.drop("lung_cancer", axis=1)
y = df["lung_cancer"]

"""
# ============================================
# REMOVE OUTLIERS (optional - does not rly immprove things)
# ============================================
age_min = 29
age_max = 80

df = df[df['age'].between(age_min, age_max)]
"""
# ============================================
# 2. SPLIT BEFORE OVERSAMPLING  (CORRECT!!!)
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

ros = RandomOverSampler(random_state=42)
X_train_over, y_train_over = ros.fit_resample(X_train, y_train)

print("Train after oversampling :", X_train_over.shape)
print("Test set remains unchanged:", X_test.shape)

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

results = []

for name, model in models.items():
    # Cross-validated recall
    cv_recall = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="recall"              # focus on recall
        # scoring="recall_weighted"     # weighted recall to account for class imbalance
        # scoring="recall_macro"        # macro recall to treat classes equally
    ).mean()

    # Fit on full training data
    model.fit(X_train, y_train)

    # Test performance
    y_pred = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)

    results.append({
        "Model": name,
        "CV Recall": cv_recall,
        "Test Recall": test_recall,
        "Test Accuracy": test_accuracy
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Test Recall", ascending=False)

print(results_df)
