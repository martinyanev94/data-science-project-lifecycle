import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-item interaction matrix
user_item_matrix = np.array([
    [1, 0, 0, 1],  # User 1
    [0, 1, 1, 0],  # User 2
    [1, 1, 0, 0],  # User 3
    [0, 0, 1, 1]   # User 4
])

# Calculate similarity scores
similarity = cosine_similarity(user_item_matrix)
print("User Similarity Scores:\n", similarity)

# Predict preferences for User 1
predicted_preferences = np.dot(similarity[0], user_item_matrix) / np.array([np.abs(similarity[0]).sum()])
print("Predicted Preferences for User 1:", predicted_preferences)
