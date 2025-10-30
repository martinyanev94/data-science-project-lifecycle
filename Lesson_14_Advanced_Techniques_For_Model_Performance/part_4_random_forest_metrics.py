from sklearn.metrics import classification_report, roc_auc_score

# Assuming rf_predictions is from the Random Forest example
print(classification_report(y_test, rf_predictions))

# If it's a probability prediction scenario:
rf_proba = rf.predict_proba(X_test)[:, 1]  # Probability of the positive class
auc_score = roc_auc_score(y_test, rf_proba)
print("Random Forest ROC AUC Score: ", auc_score)
