'''
This is a gaming bot for https://github.com/K2InformaticsGmbH/egambo. It will hopefully beat all the other bots
and humans with its Tensorflow Machine Learning Model!

This code is borrowed from:
Original Author: Aymeric Damien
Original Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import trainingData

mydata = trainingData.Data()


import tensorflow as tf

# Parameters
learning_rate = 0.01
training_epochs = 5
batch_size = 200
display_step = 1

# Network Architecture Parameters
n_hidden_1 = mydata.number_of_features * 3 # 1st layer number of features
n_hidden_2 = mydata.number_of_features * 3 # 2nd layer number of features
n_input = mydata.number_of_features # Input features. Count of board tales
n_classes = mydata.number_of_output_classes # The number of possible moves

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mydata.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mydata.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
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

    prediction = tf.argmax(pred, 1)
    print("Prediction: ", prediction.eval(feed_dict={x: mydata.test.boards}))

    # Save the mode
    saver = tf.train.Saver()
    tf.add_to_collection('input', x)
    tf.add_to_collection('pred', pred)
    saver.save(sess, 'tf-model/model')
    saver.export_meta_graph('tf-model/model.meta')


