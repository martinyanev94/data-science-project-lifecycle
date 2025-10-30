import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data.csv')

# Display basic statistics
print(df.describe())

# Create a correlation matrix
corr = df.corr()
sns.heatmap(corr, annot=True)

# Show plots
plt.show()
