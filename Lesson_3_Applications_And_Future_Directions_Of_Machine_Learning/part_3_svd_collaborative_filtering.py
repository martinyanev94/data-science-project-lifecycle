from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load dataset (MovieLens dataset placeholder)
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data.build_full_trainset(), test_size=0.2)

# Implementing SVD for collaborative filtering
model = SVD()
model.fit(trainset)

predictions = model.test(testset)
accuracy.rmse(predictions)
