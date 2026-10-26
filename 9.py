import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

X, _ = make_blobs(
    n_samples=100, centers=3, cluster_std=1.2, random_state=42
)

models = {
    "K-Means": KMeans(n_clusters=3, random_state=42, n_init=10),
    "Agglomerative": AgglomerativeClustering(n_clusters=3)
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, (name, model) in zip(axes, models.items()):
    labels = model.fit_predict(X)

    print(name, "Silhouette Score:",
          silhouette_score(X, labels))

    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
    ax.set_title(name)

    if name == "K-Means":
        ax.scatter(model.cluster_centers_[:, 0],
                   model.cluster_centers_[:, 1],
                   marker="X", s=200, c="red")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
dendrogram(linkage(X, method="ward"))
plt.title("Agglomerative Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()