from sklearn.linear_model import LinearRegression
import numpy as np

# Sample historical sales data
weeks = np.array([[1], [2], [3], [4], [5]])
sales = np.array([150, 200, 250, 300, 350])

# Model building and training
model = LinearRegression()
model.fit(weeks, sales)

# Predict sales for the next week
predicted_sales = model.predict([[6]])
print("Predicted sales for week 6:", predicted_sales[0])
