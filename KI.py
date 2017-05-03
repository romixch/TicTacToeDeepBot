import tensorflow as tf
import TicTacToe as t
import random
import numpy


class TicTacToeKI:

    def __init__(self, number_of_features, number_of_outputs):
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
            predictions = self._sess.run(self._variables, feed_dict={self._input_vars: [board]})[0]

            shift_moves = random.randint(0, 2)
            while shift_moves > 0:
                i = numpy.argmax(predictions)
                predictions[i] = predictions.min() - 1.0
                shift_moves -= 1

            x = 0
            y = 0
            minimum = predictions.min()
            while True:
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
