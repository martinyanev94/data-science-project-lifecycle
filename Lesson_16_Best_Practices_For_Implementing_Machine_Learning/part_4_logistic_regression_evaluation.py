from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = LogisticRegression()

# Evaluate using cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f'Cross-validated scores: {cv_scores}')
