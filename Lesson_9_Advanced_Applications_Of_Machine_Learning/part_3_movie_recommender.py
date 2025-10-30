from sklearn.neighbors import NearestNeighbors

# Sample dataset loading
ratings = pd.read_csv('movie_ratings.csv')  # Assume this dataset contains user ratings on movies

# Creating a pivot table
pivot_table = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# Fitting the model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(pivot_table.values)

# Making a recommendation for a specific user
user_index = 0  # change this to the index of the user
distances, indices = model_knn.kneighbors(pivot_table.iloc[user_index].values.reshape(1, -1), n_neighbors=6)

print("Recommendations for User:", ratings['movie_id'].iloc[indices.flatten()])
