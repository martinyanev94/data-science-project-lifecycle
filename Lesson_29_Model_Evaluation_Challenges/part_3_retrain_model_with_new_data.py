# Suppose new_data is the new incoming dataset
new_data = pd.DataFrame(...) # Collect new data from an external source

# Retrain the model with new data
X_new = new_data.drop('target', axis=1)
y_new = new_data['target']
model.fit(X_new, y_new)

# Predict on the previous test set again
y_pred_new = model.predict(X_test)

# Evaluate new accuracy
new_accuracy = accuracy_score(y_test, y_pred_new)
print(f"New Accuracy after Retraining: {new_accuracy:.2f}")
