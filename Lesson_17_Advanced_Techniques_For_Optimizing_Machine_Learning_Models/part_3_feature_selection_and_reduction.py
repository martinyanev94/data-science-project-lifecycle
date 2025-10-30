from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Initialize the model
model = LogisticRegression()

# Use RFE to select the top 5 features
selector = RFE(estimator=model, n_features_to_select=5, step=1)
selector = selector.fit(X_train, y_train)

# Get the selected features
selected_features = selector.support_
print(f'Selected Features: {selected_features}')
from sklearn.decomposition import PCA

# Initialize PCA and define the number of components
pca = PCA(n_components=2)

# Fit and transform the data
X_pca = pca.fit_transform(X)

print(f'Original shape: {X.shape}')
print(f'Reduced shape: {X_pca.shape}')
