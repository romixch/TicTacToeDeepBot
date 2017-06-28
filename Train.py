'''
This is a gaming bot for https://github.com/K2InformaticsGmbH/egambo. It will hopefully beat all the other bots
and humans with its Tensorflow Machine Learning Model!

This code is borrowed from:
Original Author: Aymeric Damien
Original Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import trainingData

mydata = trainingData.Data()

import tensorflow as tf
import numpy

# Parameters
learning_rate = 0.007
training_epochs = 50
batch_size = 1000
display_step = 1

# Network Architecture Parameters
n_input = mydata.number_of_features # Input features. Count of board tales
n_hidden_1 = mydata.number_of_features  # 1st layer number of features
n_hidden_2 = mydata.number_of_features  # 2nd layer number of features
n_classes = mydata.number_of_output_classes # The number of possible moves

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    with tf.name_scope('fully_connected_1'):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation
    with tf.name_scope('fully_connected_2'):
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)

    # Output layer with linear activation
    with tf.name_scope('output'):
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.01)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=0.01))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], stddev=0.01)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_classes], stddev=0.01))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.squared_difference(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        mydata.train.reset_batch()
        total_batch = int(mydata.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mydata.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})

            # X X
            #   XO
            #  O X
            # O  O
            board = [ord(value) for value in 'X X   XO O XO  O']

            predictions = sess.run(pred, feed_dict={x: [board]})[0]

            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy on training set:", accuracy.eval({x: mydata.train.boards, y: mydata.train.labels}))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy on test set:", accuracy.eval({x: mydata.test.boards, y: mydata.test.labels}))

    # Save the model
    saver = tf.train.Saver()
    tf.add_to_collection('input', x)
    tf.add_to_collection('pred', pred)
    saver.save(sess, 'tf-model/model')
    saver.export_meta_graph('tf-model/model.meta')


