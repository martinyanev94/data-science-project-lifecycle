import matplotlib.pyplot as plt

# Sample accuracy data
models = ['Model A', 'Model B', 'Model C']
accuracy = [0.85, 0.90, 0.78]

plt.bar(models, accuracy, color=['blue', 'green', 'red'])
plt.ylim(0, 1)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Models')
plt.show()
