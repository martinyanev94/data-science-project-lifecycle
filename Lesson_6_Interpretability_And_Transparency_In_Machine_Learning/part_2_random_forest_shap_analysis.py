import shap
from sklearn.ensemble import RandomForestClassifier

# Sample data
data = {
    'Feature1': [1, 2, 3, 4, 5, 6],
    'Feature2': [10, 20, 10, 30, 40, 50],
    'Target': [0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Defining features and target
X = df[['Feature1', 'Feature2']]
y = df['Target']

# Creating and fitting a random forest model
rf_model = RandomForestClassifier()
rf_model.fit(X, y)

# Explaining the model's predictions using SHAP
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X)

# Visualizing SHAP values for the first instance
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X.iloc[0,:])
