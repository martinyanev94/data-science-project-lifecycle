from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression

# Create a synthetic dataset
X, y = make_regression(n_samples=100, n_features=20, noise=0.1)

# Create a Lasso regression model
lasso = Lasso(alpha=0.1)

# Fit the model
lasso.fit(X, y)

# Get the coefficients
print(f"Coefficients: {lasso.coef_}")
