from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

# Sample data
data = {'House Size (sq ft)': [1500, 1800, 2400, 3000, 3500],
        'Price ($)': [200000, 250000, 350000, 450000, 500000]}
df = pd.DataFrame(data)

# Min-Max Scaling
min_max_scaler = MinMaxScaler()
df[['House Size (scaled)', 'Price (scaled)']] = min_max_scaler.fit_transform(df[['House Size (sq ft)', 'Price ($)']])

# Standard Scaling
standard_scaler = StandardScaler()
df[['House Size (std)', 'Price (std)']] = standard_scaler.fit_transform(df[['House Size (sq ft)', 'Price ($)']])

print(df)
