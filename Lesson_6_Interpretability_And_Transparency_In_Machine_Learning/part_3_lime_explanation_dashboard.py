import lime
import lime.lime_tabular

# Creating a LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X), 
    mode='classification', 
    feature_names=X.columns,
    class_names=['Denied', 'Approved'],
    discretize_continuous=True
)

# Explaining the first instance
exp = explainer.explain_instance(X.iloc[0].values, rf_model.predict_proba)
exp.show_in_notebook(show_table=True)
