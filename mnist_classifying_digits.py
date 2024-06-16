import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# Preprocess the data
train_images = x_train.reshape(60000, 784)
test_images = x_test.reshape(10000, 784)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
x_train, x_test = train_images / 255.0, test_images / 255.0

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='sgd',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=20, batch_size=100, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

# Misclassified images
predictions = model.predict(x_test)
misclassified = []
for i in range(len(x_test)):
    if np.argmax(predictions[i]) != np.argmax(y_test[i]):
        misclassified.append(i)

print("Number of misclassified images:", len(misclassified))

# Effect of hidden neurons
hidden_neurons = [256, 512, 1024]
for nodes in hidden_neurons:
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784,)),
        tf.keras.layers.Dense(nodes, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=20, batch_size=100, validation_data=(x_test, y_test))
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Hidden neurons:", nodes, "Test accuracy:", test_acc)

# Effect of learning rates
learning_rates = [0.1, 0.5, 1.0, 2.0]
for rate in learning_rates:
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=20, batch_size=100, validation_data=(x_test, y_test))
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Learning rate:", rate, "Test accuracy:", test_acc)

# Effect of another hidden layer
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='sgd',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=100, validation_data=(x_test, y_test))
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Two hidden layers Test accuracy:", test_acc)

# Effect of batch sizes
batch_sizes = [50, 100, 200]
for batch_size in batch_sizes:
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=20, batch_size=batch_size, validation_data=(x_test, y_test))
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Batch size:", batch_size, "Test accuracy:", test_acc)
