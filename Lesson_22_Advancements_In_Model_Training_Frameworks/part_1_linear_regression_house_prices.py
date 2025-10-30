import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    'rooms': [1, 2, 3, 4, 5],
    'area': [800, 1500, 2000, 2500, 3000],
    'price': [150000, 250000, 350000, 450000, 550000]
}
df = pd.DataFrame(data)

# Features and target variable
X = df[['rooms', 'area']]
y = df['price']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
predictions = model.predict(X_test)

print(predictions)
