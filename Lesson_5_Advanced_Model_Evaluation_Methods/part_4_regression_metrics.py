from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# True and predicted values for regression
y_actual = np.array([3, -0.5, 2, 7])
y_predicted = np.array([2.5, 0.0, 2, 8])

# Calculating MAE and MSE
mae = mean_absolute_error(y_actual, y_predicted)
mse = mean_squared_error(y_actual, y_predicted)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)

# Calculating R² Score
r2 = r2_score(y_actual, y_predicted)
print("R² Score:", r2)
