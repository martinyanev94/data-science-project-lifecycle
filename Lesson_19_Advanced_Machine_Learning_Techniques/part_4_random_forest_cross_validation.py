from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create a random forest classifier
model = RandomForestClassifier()

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=5)

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean()}")
