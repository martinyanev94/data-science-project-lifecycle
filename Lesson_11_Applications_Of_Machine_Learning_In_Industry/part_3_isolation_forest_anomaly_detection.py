from sklearn.ensemble import IsolationForest

# Sample data: transaction amounts (in thousands)
transactions = [[100], [150], [120], [130], [110], [2000], [125], [115], [145], [130]]

model = IsolationForest(contamination=0.1)
model.fit(transactions)

# Predicting anomalies
predictions = model.predict(transactions)
print(predictions)
