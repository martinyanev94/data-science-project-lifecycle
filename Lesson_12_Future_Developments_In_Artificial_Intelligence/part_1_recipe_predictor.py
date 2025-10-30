import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Sample dataset representing available ingredients and corresponding recipes
data = {
    'ingredient1': ['chicken', 'pasta', 'spinach', 'chicken', 'spinach'],
    'ingredient2': ['garlic', 'tomato', 'chicken', 'pasta', 'garlic'],
    'favorite_recipe': ['chicken garlic', 'pasta salad', 'spinach salad', 'chicken alfredo', 'spinach pasta']
}

df = pd.DataFrame(data)
X = df[['ingredient1', 'ingredient2']]
y = df['favorite_recipe']

# Encoding categorical features
X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predicting the favorite recipe based on available ingredients
sample_data = pd.get_dummies(pd.DataFrame({'ingredient1': ['chicken'], 'ingredient2': ['garlic']}))
predicted_recipe = model.predict(sample_data)
print(predicted_recipe)
