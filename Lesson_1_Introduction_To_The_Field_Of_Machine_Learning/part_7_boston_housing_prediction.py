from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

# Load Boston housing dataset
boston = load_boston()
X_boston = boston.data
y_boston = boston.target

# Creating the model
lr = LinearRegression()
# Fit the model
lr.fit(X_boston, y_boston)

# Making predictions
predictions = lr.predict(X_boston)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_boston, predictions)
print(f'Mean Squared Error: {mse:.2f}')
