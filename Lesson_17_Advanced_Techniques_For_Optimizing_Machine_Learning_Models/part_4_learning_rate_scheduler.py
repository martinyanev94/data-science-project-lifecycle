from tensorflow.keras.callbacks import LearningRateScheduler

# Define a learning rate schedule
def scheduler(epoch, lr):
    if epoch > 10:
        return lr * 0.9  # Reduce learning rate by 10%
    return lr

# Initialize the callback
lr_scheduler = LearningRateScheduler(scheduler)

# Fit your model with the learning rate scheduler
model.fit(X_train, y_train, epochs=50, callbacks=[lr_scheduler])
