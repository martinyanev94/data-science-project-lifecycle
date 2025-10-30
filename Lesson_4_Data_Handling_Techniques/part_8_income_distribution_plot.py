import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the income distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Income'], bins=10, kde=True)
plt.title('Income Distribution')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.show()
