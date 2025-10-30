# Creating a second DataFrame
income_data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Income': [50000, 60000, 70000, 80000]
}

income_df = pd.DataFrame(income_data)

# Merging both DataFrames
merged_df = pd.merge(df, income_df, on='Name', how='outer')

print("\nMerged DataFrame:")
print(merged_df)
