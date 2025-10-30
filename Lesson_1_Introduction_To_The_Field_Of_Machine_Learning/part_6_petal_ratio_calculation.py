# Creating new feature: petal length to petal width ratio
petal_ratio = X[:, 2] / X[:, 3]
X_new = np.column_stack((X, petal_ratio))
