import pandas as pd

# Sample data for features and target variable
data = {
    'Feature 1': np.random.uniform(1, 10, size=100),
    'Feature 2': np.random.uniform(1, 10, size=100),
    'Target': np.random.choice(['A', 'B'], size=100)
}
df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Feature 1', y='Feature 2', hue='Target', style='Target', palette='deep')
plt.title('Scatter Plot of Features')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='upper left')
plt.show()
