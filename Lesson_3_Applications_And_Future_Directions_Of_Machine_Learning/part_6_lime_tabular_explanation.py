from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns.tolist(), class_names=['Class 0', 'Class 1'], discretize_continuous=True)
i = 0  # index of the instance you want to explain
exp = explainer.explain_instance(X_test.values[i], model.predict_proba)
exp.show_in_notebook(show_table=True)
