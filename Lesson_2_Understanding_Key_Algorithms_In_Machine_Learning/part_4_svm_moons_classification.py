from sklearn import datasets
from sklearn.svm import SVC

# Load dataset
X, y = datasets.make_moons(n_samples=100, noise=0.1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Support Vector Classifier
svm_model = SVC(kernel='rbf', gamma='scale')
svm_model.fit(X_train, y_train)
accuracy = svm_model.score(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')
def plot_svm_decision_boundary(model, X, y):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title('SVM Decision Boundary')
    plt.show()

plot_svm_decision_boundary(svm_model, X, y)
