from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

# Load the dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)

# Create a linear regression model
model = LinearRegression()

# Create RFE model and select top 5 features
rfe = RFE(model, 5)
fit = rfe.fit(X, y)

# Print the features ranking
print(f'Top features: {X.columns[fit.support_]}')
