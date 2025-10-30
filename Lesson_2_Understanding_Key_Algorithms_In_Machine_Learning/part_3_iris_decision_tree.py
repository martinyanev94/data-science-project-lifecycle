from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create decision tree classifier
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
from sklearn.tree import plot_tree

plt.figure(figsize=(12, 8))
plot_tree(tree_model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title('Decision Tree Visualization')
plt.show()
