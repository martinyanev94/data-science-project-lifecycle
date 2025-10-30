import pandas as pd
from datetime import datetime

# Example dataframe with timestamp data
data = {'timestamp': ['2023-01-01 08:30:00', '2023-01-01 09:00:00', '2023-01-02 10:15:00']}
df = pd.DataFrame(data)

# Convert the timestamp column to datetime objects
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create new features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.day_name()
df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5

print(df)
