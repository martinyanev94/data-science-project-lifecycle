from sklearn.tree import DecisionTreeClassifier

# Sample operational data: features might include temperature, vibration, and operational hours
data = {
    'temperature': [200, 195, 300, 400, 199],
    'vibration': [5, 5, 10, 15, 5],
    'operating_hours': [100, 200, 300, 400, 150],
    'failure': [0, 0, 1, 1, 0]
}

df = pd.DataFrame(data)
X = df[['temperature', 'vibration', 'operating_hours']]
y = df['failure']

model = DecisionTreeClassifier()
model.fit(X, y)

# Making predictions
predictions = model.predict([[250, 6, 250]])
print(predictions)
