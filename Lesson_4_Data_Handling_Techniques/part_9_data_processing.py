df.to_csv('processed_data.csv', index=False)
loaded_df = pd.read_csv('processed_data.csv')
