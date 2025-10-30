from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
scaler = MinMaxScaler()

# Fitting and transforming the numeric column
df[['Age']] = scaler.fit_transform(df[['Age']])

print("\nScaled DataFrame:")
print(df)
