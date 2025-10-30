from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_groups import ClassificationPerformance

# Column mapping for comparison
column_mapping = ColumnMapping(
    target="target_column",
    prediction="prediction_column",
)

# Construct and generate the report
report = Report(metrics=[
    ClassificationPerformance(),
])

# Save the report to HTML to analyze later
report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)
report.save_html("model_performance_report.html")
