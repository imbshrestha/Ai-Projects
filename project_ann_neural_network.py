"""
Hand-Made Shallow ANN in Python

This Python script implements a basic 2-layer Artificial Neural Network (ANN) using static backpropagation
and Numpy. The ANN is designed to perform a basic function, such as guessing the next number in a series.
It includes an input layer, a hidden layer, and an output layer with weights between them.

Features:
- Input layer processes input data as a matrix and passes it on.
- Hidden layer applies a deliberate activation function.
- Output layer returns the final output.
- Weights between layers are updated through gradient descent based on a chosen loss function.

Functions:
- Multiply the input by a set of weights via matrix multiplication.
- Apply a deliberate activation function for every hidden layer.
- Return the final output.
- Calculate error by taking the difference from the desired output and the predicted output, providing the gradient descent for the loss function.
- Apply the loss function to weights.
- Repeat the training process for a minimum of 1,000 epochs.

Usage:
Run the script (with the .py extension) and input a series of two digits when prompted (e.g., '0 1').
The script will predict and visualize the final variable in the input set.
"""

import numpy as np
import matplotlib.pyplot as plt


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)


# Generate training data
input_data = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])

# Desired output for a simple pattern (adding the first and second variables)
desired_output = np.array([[0], [1], [1], [2]])

# Set seed for reproducibility
np.random.seed(42)

# Initialize weights and biases
input_size = 2
hidden_size = 4
output_size = 1

# Randomly initialize weights for the connections between layers
weights_input_hidden = np.random.rand(input_size, hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)

# Learning rate and number of training epochs
learning_rate = 0.1
epochs = 10000

# Training the neural network
for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(input_data, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)

    # Calculate error
    error = desired_output - predicted_output

    # Backpropagation
    output_error = error * sigmoid_derivative(predicted_output)
    hidden_layer_error = output_error.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)

    # Update weights using gradient descent
    weights_hidden_output += hidden_layer_output.T.dot(output_error) * learning_rate
    weights_input_hidden += input_data.T.dot(hidden_layer_error) * learning_rate

# User-friendly activation
user_input = input("Enter a series of two digits separated by a space (e.g., '0 1'): ")
user_input = np.array([list(map(int, user_input.split()))])

# Predict the next variable in the series
hidden_layer_prediction = sigmoid(np.dot(user_input, weights_input_hidden))
predicted_final_variable = sigmoid(np.dot(hidden_layer_prediction, weights_hidden_output))

# Visualize the prediction alongside the actual series
plt.plot(desired_output, label='Actual Series')
plt.scatter(len(desired_output), predicted_final_variable, color='red', label='Predicted Next Variable')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()
