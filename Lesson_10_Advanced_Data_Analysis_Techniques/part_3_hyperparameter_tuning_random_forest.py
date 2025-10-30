from sklearn.model_selection import GridSearchCV

# Define model and hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                           param_grid=param_grid, 
                           scoring='accuracy', 
                           cv=3)

# Fit grid search
grid_search.fit(X_train, y_train)

# View best hyperparameters
print(f'Best Hyperparameters: {grid_search.best_params_}')
