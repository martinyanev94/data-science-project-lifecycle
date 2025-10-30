import seaborn as sns
import numpy as np

# Sample prediction errors
np.random.seed(0)
errors = np.random.normal(size=100)

sns.boxplot(data=errors)
plt.title('Box Plot of Prediction Errors')
plt.ylabel('Error')
plt.show()
