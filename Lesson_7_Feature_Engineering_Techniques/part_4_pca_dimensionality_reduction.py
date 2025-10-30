from sklearn.decomposition import PCA

# Assume df is your dataset
pca = PCA(n_components=2)  # We want to reduce our data to 2 dimensions
X_pca = pca.fit_transform(X)

# Convert PCA result to DataFrame
pca_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
print(pca_df)
