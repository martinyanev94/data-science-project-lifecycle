import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Sample data preparation
data = {
    'temperature': [30, 32, 34, 33, 28, 31, 29, 35],
    'humidity': [60, 65, 70, 68, 55, 62, 58, 72],
    'flu_cases': [0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Splitting the dataset into features and target
X = df[['temperature', 'humidity']]
y = df['flu_cases']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Building and training the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
print("Predictions on test set:", predictions)
