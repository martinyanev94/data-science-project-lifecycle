import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Simulated dataset with sensitive attributes
data = {
    'age': [22, 45, 34, 25, 50],
    'income': [45000, 70000, 60000, 52000, 80000],
    'gender': ['female', 'male', 'female', 'male', 'female'],
    'loan_approved': [0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)
X = df[['age', 'income']]
y = df['loan_approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
