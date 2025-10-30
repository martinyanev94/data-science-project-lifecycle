from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create a logistic regression model
model = LogisticRegression(max_iter=200)

# Create RFE model and select top 2 features
rfe = RFE(model, 2)
fit = rfe.fit(X, y)

print("Selected features:\n", fit.support_)
print("Feature ranking:\n", fit.ranking_)
