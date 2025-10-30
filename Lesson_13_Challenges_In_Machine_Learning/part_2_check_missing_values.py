# Checking for missing values
missing_values = merged_data.isnull().sum()
print(missing_values)

# Dropping rows with missing target values
cleaned_data = merged_data.dropna(subset=['target'])
