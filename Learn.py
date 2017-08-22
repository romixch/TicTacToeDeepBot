import tensorflow as tf
import pickle
import model_helper
import config
import TicTacToe as ttt
import numpy


# Build model
b = ttt.TicTacToe(config.board_size, config.board_size, config.runlength)
x, y, pred, x_wins, o_wins, draw = model_helper.build_model(b.board_field_size, config.hidden_layers, config.hidden_layer_size_factor)

# Define loss and optimizer
cost_func = tf.reduce_mean(tf.squared_difference(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(cost_func)

# Initializing the variables
init = tf.global_variables_initializer()


# Learn for X
with open('learn_x.dat', 'rb') as learn_x_file:
    states = []
    rewards = []

    print('Reading games...')
    while True:
        try:
            rows = pickle.load(learn_x_file).tolist()
            for row in rows:
                states.append(row)
            rows = pickle.load(learn_x_file).tolist()
            for row in rows:
                rewards.append(row)
        except EOFError:
            break

    # width = config.board_size * config.board_size
    # states = numpy.reshape(states, newshape=(int(len(states) / width), width))
    # rewards = numpy.reshape(rewards, newshape=(int(len(rewards) / width), width))

    with tf.Session() as sess:
        sess.run(init)

        print('Learning...')
        for epoch in range(config.learning_epochs):
            _, cost = sess.run([optimizer, cost_func], feed_dict={x: states, y: rewards})
            print('Learning for X: Epoch ', epoch, ' of ', config.learning_epochs, ' with cost of ', cost)

        print('saving model X...')
        model_helper.save_model(sess, 'tf-model/X', x, pred)


# Learn for O
with open('learn_o.dat', 'rb') as learn_o_file:
    states = []
    rewards = []

    print('Reading games...')
    while True:
        try:
            rows = pickle.load(learn_o_file).tolist()
            for row in rows:
                states.append(row)
            rows = pickle.load(learn_o_file).tolist()
            for row in rows:
                rewards.append(row)
        except EOFError:
            break

    with tf.Session() as sess:
        sess.run(init)

        print('Learning...')
        for epoch in range(config.learning_epochs):
            _, cost = sess.run([optimizer, cost_func], feed_dict={x: states, y: rewards})
            print('Learning for O: Epoch ', epoch, ' of ', config.learning_epochs, ' with cost of ', cost)

        print('saving model O...')
        model_helper.save_model(sess, 'tf-model/O', x, pred)