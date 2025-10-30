from sklearn.model_selection import RandomizedSearchCV

# Define parameter distributions
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
}

random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, scoring='f1', cv=3)
random_search.fit(X_train, y_train)

print(f"Best Parameters from Randomized Search: {random_search.best_params_}")
