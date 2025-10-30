from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

# Initialize a logistic regression model
model = LogisticRegression()

# Recursive feature elimination with cross-validation
selector = RFECV(estimator=model, step=1, cv=5)
selector = selector.fit(X_train, y_train)

# Best features
selected_features = selector.support_
print("Selected Features:", X_train.columns[selected_features])
