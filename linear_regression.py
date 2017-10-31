import tensorflow as tf
import numpy as np

# Generate samples of a function we are trying to predict:
samples = 100
xs = np.linspace(-5, 5, samples)
# We will attempt to fit this function
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, samples)

# First, create TensorFlow placeholders for input data (xs) and
# output (ys) data. Placeholders are inputs to the computation graph.
# When we run the graph, we need to feed values for the placerholders into the graph.
# TODO: create placeholders for inputs and outputs
xs_p = tf.placeholder(tf.float32)
ys_p = tf.placeholder(tf.float32)

# We will try minimzing the mean squared error between our predictions and the
# output. Our predictions will take the form X*W + b, where X is input data,
# W are ou weights, and b is a bias term:
# minimize ||(X*w + b) - y||^2
# To do so, you will need to create some variables for W and b. Variables
# need to be initialised; often a normal distribution is used for this.
# TODO create weight and bias variables
W = tf.Variable(np.random.normal(size=[]), dtype=tf.float32)
b = tf.Variable(np.random.normal(size=[]), dtype=tf.float32)

# Next, you need to create a node in the graph combining the variables to predict
# the output: Y = X * w + b. Find the appropriate TensorFlow operations to do so.
predictions = xs_p * W + b
# TODO prediction

# Finally, we need to define a loss that can be minimized using gradient descent:
# The loss should be the mean squared difference between predictions
# and outputs.
diff = predictions - ys_p
sq_diff = tf.square(diff)
loss = tf.reduce_sum(sq_diff)
# TODO create loss

# Use gradient descent to optimize your variables
learning_rate = 0.001
optimize_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# We create a session to use the graph and initialize all variables
session = tf.Session()
session.run(tf.global_variables_initializer())

# Optimisation loop
epochs = 1000
previous_loss = 0.0
for epoch in range(epochs):
    for (inputs, outputs) in zip(xs, ys):
        #TODO run the optimize op
        session.run(optimize_op, {xs_p : inputs, ys_p : outputs})

    # TODO compute the current loss by running the loss operation with the
    # required inputs
    current_loss = session.run(loss, {xs_p : xs, ys_p : ys})
    print('Training cost = {}'.format(current_loss))

    # Termination condition for the optimization loop
    if np.abs(previous_loss - current_loss) < 0.000001:
        break

    previous_loss = current_loss

writer = tf.summary.FileWriter('tmp', graph=tf.get_default_graph())

# TODO try plotting the predictions by using the model to predict outputs, e.g.:
import matplotlib.pyplot as plt
predictions = session.run(predictions, {xs_p : xs})
plt.plot(xs, predictions)
plt.plot(xs, ys)
plt.show()
