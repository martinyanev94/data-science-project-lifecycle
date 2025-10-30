import horovod.tensorflow as hvd
import tensorflow as tf

# Initialize Horovod
hvd.init()

# Load the dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, 10)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model with Horovod's optimizer
opt = tf.keras.optimizers.Adam()
opt = hvd.DistributedOptimizer(opt)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, batch_size=64, epochs=10, callbacks=[hvd.callbacks.BroadcastGlobalVariablesCallback(0)])
