from sklearn.metrics import confusion_matrix
import numpy as np

# True Labels and Predicted Labels
y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 0])

# Generating the confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
