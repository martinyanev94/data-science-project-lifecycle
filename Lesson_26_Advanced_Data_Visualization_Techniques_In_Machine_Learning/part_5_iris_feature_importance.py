from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Fit Random Forest
model = RandomForestClassifier()
model.fit(X, y)

# Feature importance
importances = model.feature_importances_

# Plot feature importance
features = iris.feature_names
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), np.array(features)[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()
