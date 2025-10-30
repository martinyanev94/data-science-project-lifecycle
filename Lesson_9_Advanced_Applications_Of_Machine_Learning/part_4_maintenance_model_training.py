from sklearn.svm import SVC

# Sample dataset loading
maintenance_data = pd.read_csv('maintenance_data.csv')  # Assume this dataset contains sensor data

# Feature and target variable definition
X_maintenance = maintenance_data[['temperature', 'vibration', 'pressure']]  # Features
y_maintenance = maintenance_data['failure']  # Target variable indicating failure occurrence

# Splitting the dataset
X_train_maintenance, X_test_maintenance, y_train_maintenance, y_test_maintenance = train_test_split(X_maintenance, y_maintenance, test_size=0.2, random_state=42)

# Model training
maintenance_model = SVC(kernel='linear')
maintenance_model.fit(X_train_maintenance, y_train_maintenance)

# Predictions
y_maintenance_pred = maintenance_model.predict(X_test_maintenance)

# Evaluation
print(confusion_matrix(y_test_maintenance, y_maintenance_pred))
print(classification_report(y_test_maintenance, y_maintenance_pred))
