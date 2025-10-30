import seaborn as sns

# Visualizing the distribution of the target variable
sns.histplot(y_boston, bins=30, kde=True)
plt.xlabel('House Prices')
plt.title('Distribution of House Prices')
plt.show()
