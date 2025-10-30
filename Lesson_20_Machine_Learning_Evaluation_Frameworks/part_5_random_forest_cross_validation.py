from sklearn.model_selection import cross_val_score

# Create a new Random Forest model
model = RandomForestClassifier()

# Perform cross-validation and evaluate scores
cv_scores = cross_val_score(model, X, y, cv=5)

print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean Cross-Validation Score: {cv_scores.mean()}")
