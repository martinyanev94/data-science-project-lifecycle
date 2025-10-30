from sklearn.tree import DecisionTreeClassifier

# Sample dataset loading
fraud_data = pd.read_csv('fraud_data.csv')  # Assume this dataset contains transaction information

# Feature and target variable definition
X_fraud = fraud_data[['transaction_amount', 'transaction_location', 'user_id']]  # Features
y_fraud = fraud_data['is_fraud']  # Target variable indicating fraud occurrence

# Splitting the dataset
X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42)

# Model training
fraud_model = DecisionTreeClassifier()
fraud_model.fit(X_train_fraud, y_train_fraud)

# Predictions
y_fraud_pred = fraud_model.predict(X_test_fraud)

# Evaluation
print(confusion_matrix(y_test_fraud, y_fraud_pred))
print(classification_report(y_test_fraud, y_fraud_pred))
