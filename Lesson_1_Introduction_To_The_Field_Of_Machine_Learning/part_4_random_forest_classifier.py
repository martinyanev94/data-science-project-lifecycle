# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=100)
# Fit the model to the training data
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')
