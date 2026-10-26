import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "SVM": make_pipeline(StandardScaler(), SVC()),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=50, random_state=42
    )
}

names, scores, cv_scores = [], [], []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    cv = cross_val_score(model, X_train, y_train, cv=5)

    acc = accuracy_score(y_test, pred)
    names.append(name)
    scores.append(acc)
    cv_scores.append(cv.mean())

    print("\n", name)
    print("Accuracy:", acc)
    print("Precision:", precision_score(y_test, pred))
    print("Recall:", recall_score(y_test, pred))
    print("F1:", f1_score(y_test, pred))
    print("Cross Validation:", cv.mean())

grid = GridSearchCV(
    make_pipeline(StandardScaler(), SVC()),
    {
        "svc__C": [0.1, 1, 10],
        "svc__kernel": ["linear", "rbf"]
    },
    cv=5
)

grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)
print("Tuned Test Accuracy:", grid.score(X_test, y_test))

plt.figure(figsize=(9, 5))
x = range(len(names))
plt.bar([i - 0.2 for i in x], scores,
        width=0.4, label="Test Accuracy")
plt.bar([i + 0.2 for i in x], cv_scores,
        width=0.4, label="CV Accuracy")

plt.xticks(list(x), names)
plt.ylim(0, 1.05)
plt.ylabel("Accuracy")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.show()