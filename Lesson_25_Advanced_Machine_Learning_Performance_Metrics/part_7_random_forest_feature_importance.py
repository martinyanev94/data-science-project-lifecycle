from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Sample data
X = [[0, 0], [1, 1], [0, 1], [1, 0]]
y = [0, 1, 1, 0]

# Create and fit the model
model = RandomForestClassifier()
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_
print(f'Feature Importances: {importances}')
