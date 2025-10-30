import shap

# Create a SHAP explainer
explainer = shap.Explainer(model, X_train)

# Calculate SHAP values
shap_values = explainer(X_test)

# Visualize the feature importance
shap.summary_plot(shap_values, X_test)
