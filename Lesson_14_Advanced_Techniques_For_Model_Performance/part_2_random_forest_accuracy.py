from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Instantiate the model
rf = RandomForestClassifier(n_estimators=100)

# Fit the model
rf.fit(X_train, y_train)

# Make predictions
rf_predictions = rf.predict(X_test)

# Evaluate model performance
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy: ", rf_accuracy)
