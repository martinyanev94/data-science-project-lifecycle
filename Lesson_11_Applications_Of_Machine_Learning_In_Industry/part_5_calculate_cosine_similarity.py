import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-item rating data
data = {
    'user1': [5, 4, 0, 0, 1],
    'user2': [4, 0, 0, 4, 0],
    'user3': [0, 0, 5, 5, 0],
    'user4': [2, 3, 0, 0, 4]
}

ratings = pd.DataFrame(data, index=['item1', 'item2', 'item3', 'item4', 'item5'])
similarity = cosine_similarity(ratings.fillna(0))

print(similarity)
