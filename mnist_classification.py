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

# Set up the deep neural network
hidden_nodes = 512
input_weights = tf.Variable(tf.random.truncated_normal([784, hidden_nodes]))
input_biases = tf.Variable(tf.zeros([hidden_nodes]))
hidden_weights = tf.Variable(tf.random.truncated_normal([hidden_nodes, 10]))
hidden_biases = tf.Variable(tf.zeros([10]))

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)),
    tf.keras.layers.Dense(hidden_nodes, activation='relu', weights=[input_weights.numpy(), input_biases.numpy()]),
    tf.keras.layers.Dense(10, weights=[hidden_weights.numpy(), hidden_biases.numpy()])
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
