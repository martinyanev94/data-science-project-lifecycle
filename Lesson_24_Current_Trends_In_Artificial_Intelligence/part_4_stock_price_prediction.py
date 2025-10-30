import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load stock price data
data = pd.read_csv('stock_prices.csv')

# Preparing features and labels
X = data[['previous_price']]
y = data['current_price']

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Visualizing the predictions
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.scatter(X_test, predictions, color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Previous Price')
plt.ylabel('Current Price')
plt.legend()
plt.show()
