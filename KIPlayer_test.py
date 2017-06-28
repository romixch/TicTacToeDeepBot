import unittest
import numpy
from KIPlayer import next_move
from TicTacToe import TicTacToe

class test_next_move_as_X(unittest.TestCase):

    def test_move_X_more_attractive(self):
        game = TicTacToe()
        player = game.playerX
        class fake_session:
            def run(self, pred, feed_dict):
                # move for player X is more attractive then move for player O
                predictions = numpy.array([[0.9, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
                return predictions
        sess = fake_session()
        x, y = next_move(game, player, sess, '', '')
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_move_O_more_attractive(self):
        game = TicTacToe()
        player = game.playerX
        class fake_session:
            def run(self, pred, feed_dict):
                # move for player X is more attractive then move for player O
                predictions = numpy.array([[0.7, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
                return predictions
        sess = fake_session()
        x, y = next_move(game, player, sess, '', '')
        self.assertEqual(x, 1)
        self.assertEqual(y, 0)


class test_next_move_as_O(unittest.TestCase):
    def test_move_X_more_attractive(self):
        game = TicTacToe()
        player = game.playerO

        class fake_session:
            def run(self, pred, feed_dict):
                # move for player X is more attractive then move for player O
                predictions = numpy.array([[0.9, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
                return predictions

        sess = fake_session()
        x, y = next_move(game, player, sess, '', '')
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_move_O_more_attractive(self):
        game = TicTacToe()
        player = game.playerX

        class fake_session:
            def run(self, pred, feed_dict):
                # move for player X is more attractive then move for player O
                predictions = numpy.array([[0.7, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
                return predictions

        sess = fake_session()
        x, y = next_move(game, player, sess, '', '')
        self.assertEqual(x, 1)
        self.assertEqual(y, 0)
