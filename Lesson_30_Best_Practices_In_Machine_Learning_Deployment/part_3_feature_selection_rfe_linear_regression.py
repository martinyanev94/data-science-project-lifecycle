from sklearn.datasets import load_boston
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Load dataset
X, y = load_boston(return_X_y=True)

# Create a linear regression model
model = LinearRegression()

# RFE for feature selection
selector = RFE(model, n_features_to_select=5)
selector = selector.fit(X, y)

print("Selected features:", selector.support_)
print("Feature ranking:", selector.ranking_)
