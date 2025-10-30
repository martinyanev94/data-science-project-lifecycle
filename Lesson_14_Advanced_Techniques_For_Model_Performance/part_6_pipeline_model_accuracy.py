from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create a pipeline for scaling and modeling
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()), 
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# Fit the pipeline on training data
pipeline.fit(X_train, y_train)

# Predictions and evaluation
pipeline_predictions = pipeline.predict(X_test)
pipeline_accuracy = accuracy_score(y_test, pipeline_predictions)
print("Pipeline Model Accuracy: ", pipeline_accuracy)
