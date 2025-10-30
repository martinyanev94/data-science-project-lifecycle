# Validate with test set
final_model = grid_search.best_estimator_
final_y_pred = final_model.predict(X_test)

final_accuracy = accuracy_score(y_test, final_y_pred)
final_precision = precision_score(y_test, final_y_pred)
final_recall = recall_score(y_test, final_y_pred)
final_f1 = f1_score(y_test, final_y_pred)

print(f"Final Model Accuracy: {final_accuracy * 100:.2f}%")
print(f"Final Model Precision: {final_precision:.2f}")
print(f"Final Model Recall: {final_recall:.2f}")
print(f"Final Model F1 Score: {final_f1:.2f}")
