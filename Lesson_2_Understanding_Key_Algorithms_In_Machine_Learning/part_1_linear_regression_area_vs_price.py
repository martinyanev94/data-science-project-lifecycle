import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample data: Area vs Price
data = {'Area': [1500, 2500, 3500, 4500, 5500],
        'Price': [300000, 500000, 700000, 900000, 1100000]}
df = pd.DataFrame(data)

X = df[['Area']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.scatter(X_test, predictions, color='red', label='Predicted Prices')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price ($)')
plt.title('Linear Regression: Area vs Price')
plt.legend()
plt.show()
