# Create a new feature from existing ones
df['debt_to_income'] = df['total_debt'] / df['annual_income']
