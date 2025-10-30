from sklearn.linear_model import LinearRegression

# Sample dataset with features like temperature, rainfall, and fertilizer use
data = {
    'temperature': [20, 22, 24, 23, 21],
    'rainfall': [150, 160, 170, 165, 155],
    'fertilizer': [30, 40, 50, 45, 35],
    'yield': [30, 32, 34, 33, 31]
}
df = pd.DataFrame(data)

# Splitting the features and target
X = df[['temperature', 'rainfall', 'fertilizer']]
y = df['yield']
model = LinearRegression()
model.fit(X, y)

# Predicting yield based on new data inputs
predicted_yield = model.predict([[26, 180, 60]])
print("Predicted crop yield:", predicted_yield[0])
