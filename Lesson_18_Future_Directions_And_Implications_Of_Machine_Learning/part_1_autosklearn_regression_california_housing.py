import autosklearn.regression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load dataset
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize the Auto-sklearn Regressor
automl = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=120, per_run_time_limit=30)

# Fit the model
automl.fit(X_train, y_train)

# Make predictions
predictions = automl.predict(X_test)
