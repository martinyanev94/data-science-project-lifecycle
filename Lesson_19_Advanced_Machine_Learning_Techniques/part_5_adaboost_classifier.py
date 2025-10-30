from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
X, y = iris.data, iris.target

# Create a base model
base_model = DecisionTreeClassifier(max_depth=1)

# Create the AdaBoost model
ada_model = AdaBoostClassifier(base_model, n_estimators=100)

# Fit the model
ada_model.fit(X, y)

# Evaluate the model
predictions = ada_model.predict(X)
print(f"AdaBoost accuracy: {accuracy_score(y, predictions)}")
