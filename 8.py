import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, ConfusionMatrixDisplay
)

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=3, random_state=42
    ),
    "Naive Bayes": GaussianNB()
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, (name, model) in zip(axes, models.items()):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred, average="macro"))
    print("Recall:", recall_score(y_test, pred, average="macro"))
    print("F1:", f1_score(y_test, pred, average="macro"))

    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, ax=ax, colorbar=False
    )
    ax.set_title(name)

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 7))
plot_tree(models["Decision Tree"],
          feature_names=data.feature_names,
          class_names=data.target_names,
          filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()