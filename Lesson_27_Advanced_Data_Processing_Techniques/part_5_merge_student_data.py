# Sample DataFrame with common key for merging
data1 = {
    'StudentID': [1, 2, 3],
    'Name': ['Alice', 'Bob', 'Charlie']
}
data2 = {
    'StudentID': [1, 2, 3],
    'Grade': ['A', 'B', 'C']
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Merging two DataFrames on StudentID
merged_df = pd.merge(df1, df2, on='StudentID')
print("\nMerged DataFrame:\n", merged_df)
