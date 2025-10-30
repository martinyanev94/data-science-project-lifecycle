# Example data
data_interaction = {'Size': [800, 1200, 1500], 'Location': ['Urban', 'Suburban', 'Urban']}
df_interaction = pd.DataFrame(data_interaction)

# Create interaction feature
df_interaction['Size_Location'] = df_interaction['Size'] * df_interaction['Location'].map({'Urban': 1, 'Suburban': 0.5})

print(df_interaction)
