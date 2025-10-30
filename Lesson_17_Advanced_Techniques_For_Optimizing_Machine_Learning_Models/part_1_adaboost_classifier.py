from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Initialize a base classifier (a decision tree)
base_classifier = DecisionTreeClassifier(max_depth=1)

# Initialize the AdaBoost classifier
ada_model = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=50, random_state=42)

# Train the model
ada_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = ada_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'AdaBoost Model Accuracy: {accuracy:.2f}')
