from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Using a Random Forest classifier for demonstration
model = RandomForestClassifier()

# Perform 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)

print("Cross-validation scores:", scores)
print("Mean CV score:", np.mean(scores))
