# Adding another column for Income
df['Income'] = [50000, None, 70000, 60000]

# Grouping by City and getting average income
avg_income = df.groupby('City')['Income'].mean()

print("\nAverage Income by City:")
print(avg_income)
