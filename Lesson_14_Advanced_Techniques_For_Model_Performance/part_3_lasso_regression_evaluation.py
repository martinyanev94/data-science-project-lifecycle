from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# Create a Lasso regression model
lasso = Lasso(alpha=0.1)

# Fit the model
lasso.fit(X_train, y_train)

# Make predictions
lasso_predictions = lasso.predict(X_test)

# Evaluate model performance
lasso_mse = mean_squared_error(y_test, lasso_predictions)
print("Lasso Regression Mean Squared Error: ", lasso_mse)
