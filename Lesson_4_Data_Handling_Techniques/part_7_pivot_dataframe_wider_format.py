# Pivoting the DataFrame into a wider format
pivot_df = df.pivot(index='City', columns='Name', values='Age')

print("\nPivoted DataFrame:")
print(pivot_df)
