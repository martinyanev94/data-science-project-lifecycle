from sklearn.ensemble import IsolationForest
import pandas as pd

# Load data
data = pd.read_csv('network_traffic.csv')

# Initialize the model
model = IsolationForest(contamination=0.01)

# Fit the model
model.fit(data)

# Predict anomalies
anomalies = model.predict(data)

# Identify outliers
data['anomaly'] = anomalies
print(data[data['anomaly'] == -1])
