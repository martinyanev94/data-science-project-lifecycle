from sklearn.metrics import mean_absolute_error, mean_squared_error

# Sample predicted and actual values
y_actual = [3.0, -0.5, 2.0, 7.0]
y_predicted = [2.5, 0.0, 2.0, 8.0]

mae = mean_absolute_error(y_actual, y_predicted)
mse = mean_squared_error(y_actual, y_predicted)

print(f'Mean Absolute Error: {mae:.2f}, Mean Squared Error: {mse:.2f}')
