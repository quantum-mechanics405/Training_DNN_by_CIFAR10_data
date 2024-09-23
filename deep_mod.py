import tensorflow as tf
import os

# Set random seed for reproducibility
tf.random.set_seed(42)

# Build the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[32, 32, 3]))
for _ in range(20):
    model.add(tf.keras.layers.Dense(100,
                                    activation="swish",
                                    kernel_initializer="he_normal"))

model.add(tf.keras.layers.Dense(10, activation="softmax"))

# Compile the model with Nadam optimizer
optimizer = tf.keras.optimizers.Nadam(learning_rate=5e-5)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])

# Load and split the CIFAR-10 dataset
cifar10 = tf.keras.datasets.cifar10.load_data()
(X_train_full, y_train_full), (X_test, y_test) = cifar10

# Normalize the pixel values to [0, 1]
X_train_full, X_test = X_train_full / 255.0, X_test / 255.0

# Split into training and validation sets
X_train = X_train_full[5000:]
y_train = y_train_full[5000:]
X_valid = X_train_full[:5000]
y_valid = y_train_full[:5000]

# Define callbacks
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("my_cifar10_model", save_best_only=True)

# Correct the log directory path
run_index = 2  # Increment every time you train the model
run_logdir = os.path.join("my_cifar10_logs_for", f"run_{run_index:03d}")
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

# Combine the callbacks
callbacks = [early_stopping_cb, model_checkpoint_cb, tensorboard_cb]

# Train the model
model.fit(X_train, y_train, epochs=10,
          validation_data=(X_valid, y_valid),
          callbacks=callbacks)
