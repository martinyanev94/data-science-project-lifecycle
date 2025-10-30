from sklearn.metrics import classification_report

# Multi-class true labels and predicted labels
y_true_multi = [0, 1, 2, 0, 1, 2]
y_pred_multi = [0, 2, 1, 0, 0, 1]

report = classification_report(y_true_multi, y_pred_multi, output_dict=True)
macro_precision = report['macro avg']['precision']
macro_recall = report['macro avg']['recall']
macro_f1 = report['macro avg']['f1-score']

print(f'Macro Precision: {macro_precision:.2f}, Macro Recall: {macro_recall:.2f}, Macro F1: {macro_f1:.2f}')
