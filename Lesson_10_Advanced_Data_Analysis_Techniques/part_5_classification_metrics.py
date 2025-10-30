from sklearn.metrics import classification_report, confusion_matrix

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

# Generate classification report
class_report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{class_report}')
