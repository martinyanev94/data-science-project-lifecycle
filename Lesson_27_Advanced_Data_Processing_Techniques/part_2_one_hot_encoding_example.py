# Sample DataFrame with categorical data
data = {
    'City': ['New York', 'Los Angeles', 'New York', 'San Francisco'],
    'Sales': [250, 150, 300, 200]
}

df = pd.DataFrame(data)
print("Original DataFrame:\n", df)

# Transforming categorical variable using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['City'])
print("\nDataFrame after One-Hot Encoding:\n", df_encoded)
