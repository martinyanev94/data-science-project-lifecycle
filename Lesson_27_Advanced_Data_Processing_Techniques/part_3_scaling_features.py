from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Sample data
data = {
    'Feature1': [100, 200, 300, 400],
    'Feature2': [0.1, 0.2, 0.3, 0.4]
}

df = pd.DataFrame(data)

# Scaling with StandardScaler
standard_scaler = StandardScaler()
df_standard_scaled = standard_scaler.fit_transform(df)
print("\nData after Standard Scaling:\n", df_standard_scaled)

# Scaling with MinMaxScaler
min_max_scaler = MinMaxScaler()
df_minmax_scaled = min_max_scaler.fit_transform(df)
print("\nData after MinMax Scaling:\n", df_minmax_scaled)
