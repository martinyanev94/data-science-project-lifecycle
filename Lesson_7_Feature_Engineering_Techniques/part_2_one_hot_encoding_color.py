# Sample data
data_color = {'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red']}
df_color = pd.DataFrame(data_color)

# One-hot encoding
df_color_encoded = pd.get_dummies(df_color, columns=['Color'], drop_first=True)

print(df_color_encoded)
