from sklearn.decomposition import PCA

# Example dataset with multiple features
X = [[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0]]

# Applying PCA
pca = PCA(n_components=1)
X_reduced = pca.fit_transform(X)

print('Reduced dimensions:', X_reduced)
