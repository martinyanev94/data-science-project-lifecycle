from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset

# Create a BinaryLabelDataset
dataset = BinaryLabelDataset(favorable_label=1, unfavorable_label=0,
                              df=df, label_names=['label'], protected_attribute_names=['gender'])

# Compute metric for fairness
metric = BinaryLabelDatasetMetric(dataset, privileged_groups=[{‘gender’: 1}],
                                   unprivileged_groups=[{‘gender’: 0}])
print(f"Disparate Impact: {metric.disparate_impact()}")
