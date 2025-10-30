from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Sample data
import numpy as np
X = np.array([[np.nan, 1], [2, 3], [np.nan, 5], [6, 7]])
y = np.array([1, 0, 1, 0])

# Pipelines for handling data imputation, scaling, and model training
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Training the model
pipeline.fit(X, y)

# Making predictions
predictions = pipeline.predict(X)
print(predictions)
