# Convert 'Age' to integer type
df['Age'] = df['Age'].astype(int)

# Display data types
print("\nData Types:")
print(df.dtypes)
