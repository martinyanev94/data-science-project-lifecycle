from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Prepare features and labels
X = cleaned_data[['feature1', 'feature2', 'feature3']]
y = cleaned_data['target']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a Support Vector Classifier
model = SVC()
model.fit(X_train, y_train)

# Predicting
predictions = model.predict(X_test)
