import tensorflow as tf
import TicTacToe as t
import random
import numpy


class TicTacToeKI:

    def __init__(self, number_of_features, number_of_outputs):
        # self._variables, self._x = setup_model(number_of_features, number_of_outputs)
        self._sess = tf.Session()
        try:
            new_saver = tf.train.import_meta_graph('tf-model/model.meta')
            new_saver.restore(self._sess, 'tf-model/model')
            self._input_vars = tf.get_collection('input')[0]
            self._variables = tf.get_collection('pred')[0]
            self._has_model = True
        except:
            self._has_model = False


    @property
    def variables(self):
        return self._variables

    def play_move(self, ttt, player):
        if self._has_model:
            board = ttt.board_for_learning(player)
            #prediction = tf.argmax(self._variables, 1)
            #prediction.eval(feed_dict={self._x: [board]}, session = self._sess)

            predictions = self._sess.run(self._variables, feed_dict={self._input_vars: [board]})[0]

            x = 0
            y = 0
            minimum = predictions.min()
            while predictions.size > 0:
                i = numpy.argmax(predictions)
                x = i % 3
                y = i / 3
                if ttt.isFieldAvailable(x, y):
                    break
                predictions[i] = minimum - 1.0

            return x, y
        else:
            while True:
                x = random.randint(0, 2)
                y = random.randint(0, 2)
                if ttt.isFieldAvailable(x, y):
                    break
        return x, y

def setup_model(number_of_features, number_of_outputs):
    # Network Architecture Parameters
    n_hidden_1 = number_of_features * 3  # 1st layer number of features
    n_hidden_2 = number_of_features * 3  # 2nd layer number of features
    n_input = number_of_features  # Input features. Count of board tales
    n_classes = number_of_outputs  # The number of possible moves

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

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
    return multilayer_perceptron(x, weights, biases), x

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
