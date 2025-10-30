import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'age': [25, 45, 65, 35, 50, 72, 68, 40],
    'previous_conditions': [1, 0, 1, 0, 1, 1, 0, 0],
    'readmitted': [0, 1, 1, 0, 1, 1, 0, 0]
}

df = pd.DataFrame(data)
X = df[['age', 'previous_conditions']]
y = df['readmitted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
