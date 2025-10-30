import pandas as pd
import numpy as np

# Creating a sample DataFrame
data = {
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [10, 11, 12, 13]
}

df = pd.DataFrame(data)
print("Original DataFrame:\n", df)

# Checking for missing values
print("\nMissing values in each column:\n", df.isnull().sum())

# Filling missing values
df['A'].fillna(df['A'].mean(), inplace=True)  # Filling with mean
df['B'].fillna(method='ffill', inplace=True)  # Forward fill

print("\nDataFrame after filling missing values:\n", df)
