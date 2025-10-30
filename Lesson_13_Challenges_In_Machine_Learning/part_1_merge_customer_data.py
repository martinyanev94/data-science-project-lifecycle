import pandas as pd

# Load multiple datasets
activity_data = pd.read_csv('activity_logs.csv')
support_data = pd.read_csv('support_logs.csv')

# Merging based on a common identifier (customer ID)
merged_data = pd.merge(activity_data, support_data, on='customer_id', how='inner')

# Displaying the first few rows of merged dataset
print(merged_data.head())
