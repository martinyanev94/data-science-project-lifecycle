from sklearn.linear_model import Ridge

# Using Ridge regression to prevent overfitting
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(X_test)

ridge_accuracy = accuracy_score(y_test, ridge_predictions)
print(f"Ridge Regression Accuracy: {ridge_accuracy:.2f}")
