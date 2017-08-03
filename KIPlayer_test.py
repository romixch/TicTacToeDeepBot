import unittest
import numpy
from KIPlayer import next_move
from TicTacToe import TicTacToe

class test_next_move(unittest.TestCase):

    def test_move_1(self):
        game = TicTacToe()
        class fake_session:
            def run(self, pred, feed_dict):
                # move for player X is more attractive then move for player O
                predictions = numpy.array([[0.9, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
                return predictions
        sess = fake_session()
        x, y = next_move(game, sess, '', '')
        self.assertEqual(int(x), 0)
        self.assertEqual(int(y), 0)

    def test_move_2(self):
        game = TicTacToe()
        class fake_session:
            def run(self, pred, feed_dict):
                # move for player X is more attractive then move for player O
                predictions = numpy.array([[0.3, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.5]])
                return predictions
        sess = fake_session()
        x, y = next_move(game, sess, '', '')
        self.assertEqual(x, 1)
        self.assertEqual(y, 2)

    def test_exploration(self):
        game = TicTacToe()
        class fake_session:
            def run(self, pred, feed_dict):
                # move for player X is more attractive then move for player O
                predictions = numpy.array([[0.3, 0.1, 0.5, 0.5, 0.5, 0.5, 0.6, 0.7, 0.5]])
                return predictions
        sess = fake_session()
        correct = next_move(game, sess, '', '')
        correct_count = 0
        for i in range(20):
            x, y = next_move(game, sess, '', '', 0.15)
            if (x == 1 and y == 2):
                correct_count += 1

        self.assertTrue(correct_count > 5, 'There was some randomness and in less than 6 out of 10 times it the result was random...')