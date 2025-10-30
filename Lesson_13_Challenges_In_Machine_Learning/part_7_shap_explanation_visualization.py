import shap

# Assuming 'model' is the trained classifier
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Visualize the first prediction's explanation
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
