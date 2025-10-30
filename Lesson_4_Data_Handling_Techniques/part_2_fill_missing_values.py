import pandas as pd

# Sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', None],
    'Age': [25, None, 30, 22],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}

df = pd.DataFrame(data)

# Display original DataFrame
print("Original DataFrame:")
print(df)

# Fill missing values for 'Age' with the average age
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Fill missing names with a placeholder
df['Name'].fillna("Unknown", inplace=True)

# Display modified DataFrame
print("\nModified DataFrame:")
print(df)
