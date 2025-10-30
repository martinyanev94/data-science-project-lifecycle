from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Define the model
model = RandomForestClassifier()

# Set the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Configure GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           scoring='accuracy', cv=5)

# Fit the model
grid_search.fit(X, y)

# Print the best parameters
print(f"Best parameters found: {grid_search.best_params_}")

# Evaluate the model
best_model = grid_search.best_estimator_
predictions = best_model.predict(X)
print(f"Accuracy: {accuracy_score(y, predictions)}")
