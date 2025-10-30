from sklearn.model_selection import cross_val_score

# Using the previously defined Random Forest model
scores = cross_val_score(rf, X_train, y_train, cv=5)

print("Cross-Validation Scores: ", scores)
print("Average Cross-Validation Score: ", scores.mean())
