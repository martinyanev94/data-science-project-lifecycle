from sklearn.ensemble import StackingClassifier

# Define base learners
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('svm', SVC(probability=True))
]

# Define the meta-learner
meta_learner = LogisticRegression()

# Create the stacking model
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner)

# Train the stacking model
stacking_model.fit(X_train, y_train)

# Evaluate the model
y_pred = stacking_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Stacking Model Accuracy: {accuracy:.2f}')
