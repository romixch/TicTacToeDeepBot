import unittest
import numpy
from TicTacToe import TicTacToe

class TicTacToeTests(unittest.TestCase):

    def test_first_case(self):
        game = TicTacToe()
        game.playX(0, 0)
        game.playX(1, 0)
        game.playX(2, 0)
        board_as_x = game.board_for_learning_as_X()
        expected_as_x = numpy.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertTrue((board_as_x==expected_as_x).all())
        board_as_o = game.board_for_learning_as_O()
        expected_as_o = numpy.array([2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertTrue((board_as_o==expected_as_o).all())


    def test_second_case(self):
        game = TicTacToe()
        game.playX(0, 0)
        game.playO(1, 1)
        game.playX(2, 2)
        board_as_x = game.board_for_learning()
        expected_as_x = numpy.array([1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0])
        self.assertTrue((board_as_x==expected_as_x).all())
        board_as_o = game.board_for_learning_as_O()
        expected_as_o = numpy.array([2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0])
        self.assertTrue((board_as_o==expected_as_o).all())