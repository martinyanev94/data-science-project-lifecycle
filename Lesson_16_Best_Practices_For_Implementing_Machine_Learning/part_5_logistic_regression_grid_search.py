from sklearn.model_selection import GridSearchCV

# Define a parameter grid
param_grid = {'C': [0.1, 1, 10], 'solver': ['newton-cg', 'lbfgs', 'liblinear']}

# Instantiate the GridSearchCV
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)

# Fit the model
grid_search.fit(X_train, y_train)
print(f'Best parameters: {grid_search.best_params_}')
