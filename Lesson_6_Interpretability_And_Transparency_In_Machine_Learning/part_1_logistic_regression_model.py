import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# Sample data
data = {
    'Income': [50000, 60000, 80000, 20000, 70000],
    'Credit_Score': [700, 720, 740, 580, 690],
    'Approved': [1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# Splitting the dataset
X = df[['Income', 'Credit_Score']]
y = df['Approved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitting the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Getting the coefficients
coef = model.coef_[0]
print(f"Coefficients: Income: {coef[0]}, Credit Score: {coef[1]}")
