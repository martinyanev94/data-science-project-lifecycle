import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample data: Stock prices over time
data = {
    'day': [1, 2, 3, 4, 5, 6, 7],
    'price': [100, 102, 101, 105, 107, 108, 110]
}

df = pd.DataFrame(data)
X = df[['day']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions)
