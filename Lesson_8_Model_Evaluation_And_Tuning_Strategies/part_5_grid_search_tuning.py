from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', cv=3)
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
