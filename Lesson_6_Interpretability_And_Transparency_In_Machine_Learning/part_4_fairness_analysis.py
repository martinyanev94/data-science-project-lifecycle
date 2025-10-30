from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

# Sample fair and biased dataset
data = {
    'Feature1': [1, 4, 6, 13, 8],
    'Feature2': [5, 3, 4, 6, 3],
    'Bias_Gender': [0, 1, 0, 1, 1],  # 0 - Female, 1 - Male
    'Approved': [1, 0, 1, 0, 0]
}

df = pd.DataFrame(data)
dataset = BinaryLabelDataset(df=df, label_names=['Approved'], protected_attribute_names=['Bias_Gender'])

# Calculating metrics
metric = BinaryLabelDatasetMetric(dataset, privileged_groups=[{"Bias_Gender": 0}], unprivileged_groups=[{"Bias_Gender": 1}])
disparate_impact = metric.disparate_impact()
print("Disparate Impact:", disparate_impact)
