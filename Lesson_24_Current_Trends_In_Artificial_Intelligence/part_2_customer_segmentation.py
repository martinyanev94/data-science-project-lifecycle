from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the customer data
data = pd.read_csv('customer_data.csv')

# Selecting features for clustering
X = data[['age', 'annual_income', 'spending_score']]

# Running K-means clustering
kmeans = KMeans(n_clusters=3)
data['Cluster'] = kmeans.fit_predict(X)

# Visualizing the clusters
plt.scatter(data['annual_income'], data['spending_score'], c=data['Cluster'], cmap='viridis')
plt.title('Customer Segmentation')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
