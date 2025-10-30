import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Sample dataset loading
data = pd.read_csv('healthcare_data.csv')  # Assume this dataset contains health metrics

# Feature and target variable definition
X = data[['age', 'bmi', 'blood_pressure']]  # Features
y = data['diabetes']  # Target variable indicating presence of diabetes

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
