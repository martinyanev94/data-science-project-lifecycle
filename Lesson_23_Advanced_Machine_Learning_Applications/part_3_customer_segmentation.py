import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('customer_data.csv')

# Selecting features
X = data[['purchase_frequency', 'average_order_value']]

# Initialize the model
kmeans = KMeans(n_clusters=3)

# Fit the model
kmeans.fit(X)

# Add cluster labels to the original data
data['cluster'] = kmeans.labels_

# Plotting the clusters
plt.scatter(data['purchase_frequency'], data['average_order_value'], c=data['cluster'])
plt.xlabel('Purchase Frequency')
plt.ylabel('Average Order Value')
plt.title('Customer Segmentation')
plt.show()
