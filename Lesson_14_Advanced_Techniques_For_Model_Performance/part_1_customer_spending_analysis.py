import pandas as pd

# Sample DataFrame
data = {
    'customer_id': [1, 2, 3],
    'join_date': pd.to_datetime(['2020-01-01', '2021-02-01', '2022-03-01']),
    'total_spent': [200, 400, 600]
}

df = pd.DataFrame(data)

# Create a new feature for membership duration
df['membership_duration'] = (pd.to_datetime('today') - df['join_date']).dt.days

# Create a new feature for spending categories
df['spending_category'] = pd.cut(df['total_spent'], bins=[0, 250, 500, float("inf")], labels=['Low', 'Medium', 'High'])

print(df)
