from sklearn.metrics import roc_auc_score

# Probabilities of positive class for each instance
y_proba = [0.1, 0.9, 0.4, 0.2, 0.8, 0.6, 0.3, 0.9]
roc_auc = roc_auc_score(y_true, y_proba)

print(f'ROC AUC Score: {roc_auc:.2f}')
