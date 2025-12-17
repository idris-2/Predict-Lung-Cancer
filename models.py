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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
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

def make_knn(k):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

# ============================================
# 5. CROSS-VALIDATION FOR K = 1..20
# ============================================

knn_scores = []

for k in range(1, 20):
    model = make_knn(k)
    scores = cross_val_score(
        model,
        X_train_over,
        y_train_over,
        cv=5,
        scoring="accuracy"  # you can change to "recall"
    )
    knn_scores.append(scores.mean())

print(knn_scores)

# ============================================
# 6. PLOT RESULTS
# ============================================

x_ticks = range(1, 20)

plt.plot(x_ticks, knn_scores)
plt.xticks(x_ticks)
plt.grid(True)
plt.xlabel("k (number of neighbors)")
plt.ylabel("Cross-validated accuracy")
plt.title("KNN Performance for Different k Values")
plt.show()

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

confusion_knn=confusion_matrix(y_test,knn.predict(X_test))
plt.figure(figsize=(8,8))
sns.heatmap(confusion_knn,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")

print(classification_report(y_test,knn.predict(X_test)))