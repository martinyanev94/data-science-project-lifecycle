from sklearn.ensemble import RandomForestClassifier

# Load Iris dataset again
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
accuracy = rf_model.score(X_test, y_test)
print(f'Random Forest Accuracy: {accuracy:.2f}')
importances = rf_model.feature_importances_
feature_names = iris.feature_names
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f+1}. {feature_names[indices[f]]} ({importances[indices[f]]:.4f})")
