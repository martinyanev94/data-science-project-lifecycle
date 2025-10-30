from sklearn.metrics import classification_report

# Predictions
y_pred = model.predict(X_test)

# Generate classification report
report = classification_report(y_test, y_pred)
print(report)
