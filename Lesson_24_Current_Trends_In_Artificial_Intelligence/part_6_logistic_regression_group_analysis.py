import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Create synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Check the accuracy for different groups (e.g., based on a feature)
group_1_accuracy = model.predict(X_test[X_test[:, 0] == 1]).mean()
group_2_accuracy = model.predict(X_test[X_test[:, 0] == 0]).mean()

print(f'Group 1 Accuracy: {group_1_accuracy}')
print(f'Group 2 Accuracy: {group_2_accuracy}')
