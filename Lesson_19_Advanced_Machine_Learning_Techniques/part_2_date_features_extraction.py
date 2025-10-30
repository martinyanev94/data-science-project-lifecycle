import pandas as pd

# Sample data with a 'date' column
data = {
    'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'value': [10, 20, 15]
}

df = pd.DataFrame(data)

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract features
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

print(df)
