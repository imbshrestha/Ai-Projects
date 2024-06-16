import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate random linear data
np.random.seed(101)
tf.random.set_seed(101)
x = np.linspace(0, 50, 50)
y = np.linspace(0, 50, 50)
x += np.random.uniform(-4, 4, 50)
y += np.random.uniform(-4, 4, 50)
n = len(x)

# Step 1: Plot the training data
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Training Data')
plt.show()

# Step 2: Define placeholders for input X and label Y
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Step 3: Define weights and bias
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

# Step 4: Define hyperparameters
learning_rate = 0.01
training_epochs = 1000

# Step 5: Define the hypothesis, cost function, and optimizer
y_pred = tf.add(tf.multiply(X, W), b)
cost = tf.reduce_sum(tf.pow(y_pred-Y, 2))/(2*n)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Step 6: Implement the training process inside a TensorFlow session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (_x, _y) in zip(x, y):
            sess.run(optimizer, feed_dict={X: _x, Y: _y})
        if (epoch+1) % 50 == 0:
            c = sess.run(cost, feed_dict={X: x, Y: y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                  "W=", sess.run(W), "b=", sess.run(b))

    training_cost = sess.run(cost, feed_dict={X: x, Y: y})
    weight = sess.run(W)
    bias = sess.run(b)

# Step 7: Print the results
print("Optimization Finished!")
print("Training cost=", training_cost, "W=", weight, "b=", bias)

# Step 8: Plot the fitted line on top of the original data
plt.scatter(x, y)
plt.plot(x, weight*x + bias, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitted Line')
plt.show()
