pip install tensorflow keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Generate dummy data
X_train = np.random.rand(1000, 10)
y_train = np.random.randint(2, size=(1000, 1))

# Define the model
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=10))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=10, batch_size=32)
