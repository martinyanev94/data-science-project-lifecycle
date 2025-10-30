from sklearn.model_selection import StratifiedKFold

X = ...  # Features
y = ...  # Target

skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # You can fit your model here, e.g.,
    # model.fit(X_train, y_train)
