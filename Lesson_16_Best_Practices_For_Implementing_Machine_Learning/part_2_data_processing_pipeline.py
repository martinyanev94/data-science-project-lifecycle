from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Impute missing values
imputer = SimpleImputer(strategy='mean')
df['feature_with_nans'] = imputer.fit_transform(df[['feature_with_nans']])

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['feature_1', 'feature_2']])
