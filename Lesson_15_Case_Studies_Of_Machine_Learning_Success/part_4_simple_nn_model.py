import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define a simple neural network model
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Input layer
    layers.Dense(128, activation='relu'),  # Hidden layer
    layers.Dense(10, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Assuming we have training data ready in (x_train, y_train)
# model.fit(x_train, y_train, epochs=10)
