import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Example transaction dataset
data = {
    'transaction_amount': [150, 2000, 300, 400, 50],
    'is_fraud': [0, 1, 0, 0, 1]
}
df = pd.DataFrame(data)

# Splitting the dataset into features and target
X = df[['transaction_amount']]
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building and training the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions and evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Fraud Detection Accuracy:", accuracy)
