import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Initial Accuracy: {accuracy:.2f}")
import numpy as np

# Simulating data drift by adding noise to the test set
noise = np.random.randn(*X_test.shape) * 0.5
X_test_drifted = X_test + noise

# Make predictions on the drifted test set
y_pred_drifted = model.predict(X_test_drifted)

# Evaluate the model again
accuracy_drifted = accuracy_score(y_test, y_pred_drifted)
print(f"Accuracy after Data Drift: {accuracy_drifted:.2f}")
