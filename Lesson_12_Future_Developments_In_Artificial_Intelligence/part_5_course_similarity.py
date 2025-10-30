import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-course rating dataset
data = {
    'user_1': [5, 0, 0, 4],
    'user_2': [0, 3, 0, 3],
    'user_3': [2, 5, 0, 0],
    'user_4': [0, 0, 4, 5]
}

ratings = pd.DataFrame(data, index=['course_1', 'course_2', 'course_3', 'course_4'])
similarity = cosine_similarity(ratings.fillna(0))

print(similarity)
