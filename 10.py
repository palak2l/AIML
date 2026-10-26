import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = load_iris()
X, y = data.data, data.target

X = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_new = pca.fit_transform(X)

print("Original Shape:", X.shape)
print("Reduced Shape:", X_new.shape)
print("Explained Variance:", pca.explained_variance_ratio_)
print("Total Variance:", sum(pca.explained_variance_ratio_))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

scatter = axes[0].scatter(
    X_new[:, 0], X_new[:, 1], c=y, cmap="viridis"
)
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].set_title("PCA - 2D Projection")
fig.legend(scatter.legend_elements()[0],
           data.target_names, loc="upper right")

axes[1].bar(["PC1", "PC2"], pca.explained_variance_ratio_)
axes[1].set_title("Explained Variance Ratio")
axes[1].set_ylabel("Variance Ratio")

plt.tight_layout()
plt.show()