from lime.lime_tabular import LimeTabularExplainer

# Create a LIME explainer for our data
explainer = LimeTabularExplainer(X_train.values, feature_names=X.columns, class_names=iris.target_names, mode='classification')

# Choose an instance to explain
i = 0
exp = explainer.explain_instance(X_test.values[i], model.predict_proba)

# Visualize the explanation
exp.show_in_notebook(show_table=True)
