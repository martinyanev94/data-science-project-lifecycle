from sklearn.cluster import DBSCAN
import numpy as np

# Simulated user behavior data (e.g., login times)
data = np.array([[1], [2], [1], [10], [11], [12], [2], [3]])

# Applying DBSCAN for anomaly detection
model = DBSCAN(eps=1.5, min_samples=2)
anomalies = model.fit_predict(data)

print(anomalies)
