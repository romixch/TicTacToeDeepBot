import unittest
import numpy
from TicTacToe import TicTacToe

class TicTacToeTests(unittest.TestCase):

    def test_3_by_3_horizontal(self):
        game = TicTacToe()
        game.playX(0, 0)
        self.assertFalse(game.isFinished())
        game.playX(1, 0)
        self.assertFalse(game.isFinished())
        game.playX(2, 0)
        self.assertTrue(game.isFinished())

    def test_3_by_3_horizontal_bottom(self):
        game = TicTacToe()
        game.playX(0, 2)
        self.assertFalse(game.isFinished())
        game.playX(1, 2)
        self.assertFalse(game.isFinished())
        game.playX(2, 2)
        self.assertTrue(game.isFinished())


    def test_3_by_3_vertical(self):
        game = TicTacToe()
        game.playX(0, 0)
        self.assertFalse(game.isFinished())
        game.playX(0, 1)
        self.assertFalse(game.isFinished())
        game.playX(0, 2)
        self.assertTrue(game.isFinished())


    def test_3_by_3_vertical_right(self):
        game = TicTacToe()
        game.playX(2, 0)
        self.assertFalse(game.isFinished())
        game.playX(2, 1)
        self.assertFalse(game.isFinished())
        game.playX(2, 2)
        self.assertTrue(game.isFinished())


    def test_3_by_3_diagonal_1(self):
        game = TicTacToe()
        game.playX(0, 0)
        self.assertFalse(game.isFinished())
        game.playX(1, 1)
        self.assertFalse(game.isFinished())
        game.playX(2, 2)
        self.assertTrue(game.isFinished())


    def test_3_by_3_diagonal_2(self):
        game = TicTacToe()
        game.playX(0, 2)
        self.assertFalse(game.isFinished())
        game.playX(1, 1)
        self.assertFalse(game.isFinished())
        game.playX(2, 0)
        self.assertTrue(game.isFinished())


    def test_4_by_4_horizontal(self):
        game = TicTacToe(4, 4, 3)
        game.playX(0, 0)
        self.assertFalse(game.isFinished())
        game.playX(1, 0)
        self.assertFalse(game.isFinished())
        game.playX(2, 0)
        self.assertTrue(game.isFinished())


    def test_4_by_4_vertical(self):
        game = TicTacToe(4, 4, 3)
        game.playX(2, 1)
        self.assertFalse(game.isFinished())
        game.playX(2, 2)
        self.assertFalse(game.isFinished())
        game.playX(2, 3)
        self.assertTrue(game.isFinished())


    def test_4_by_4_diagonal_1(self):
        game = TicTacToe(4, 4, 3)
        game.playX(0, 1)
        self.assertFalse(game.isFinished())
        game.playX(1, 2)
        self.assertFalse(game.isFinished())
        game.playX(2, 3)
        self.assertTrue(game.isFinished())


    def test_4_by_4_diagonal_2(self):
        game = TicTacToe(4, 4, 3)
        game.playX(2, 1)
        self.assertFalse(game.isFinished())
        game.playX(1, 2)
        self.assertFalse(game.isFinished())
        game.playX(0, 3)
        self.assertTrue(game.isFinished())


    def test_4_by_4_diagonal_3(self):
        game = TicTacToe(4, 4, 3)
        game.playX(1, 1)
        self.assertFalse(game.isFinished())
        game.playX(2, 2)
        self.assertFalse(game.isFinished())
        game.playX(3, 3)
        self.assertTrue(game.isFinished())

    def test_4_by_4_diagonal_4(self):
        game = TicTacToe(4, 4, 3)
        game.playX(0, 0)
        self.assertFalse(game.isFinished())
        game.playX(1, 1)
        self.assertFalse(game.isFinished())
        game.playX(2, 2)
        self.assertTrue(game.isFinished())


    def test_4_by_4_bug(self):
        game = TicTacToe(4, 4, 3)
        game.playX(3, 2)
        game.playO(1, 2)
        game.playX(0, 1)
        game.playO(2, 0)
        game.playX(2, 3)
        print(game.get_pretty_board)
        self.assertFalse(game.isFinished())

    def test_4_by_4_bug_2(self):
        game = TicTacToe(4, 4, 3)
        game.playX(0, 0)
        game.playO(1, 0)
        game.playX(0, 3)
        game.playO(0, 1)
        game.playX(3, 3)
        game.playO(3, 2)
        print(game.get_pretty_board)
        self.assertFalse(game.isFinished())

    def test_5_by_5_bug(self):
        game = TicTacToe(5, 5, 4)
        game.playX(0, 0)
        game.playO(2, 1)
        game.playX(1, 1)
        game.playO(2, 3)
        game.playX(2, 2)
        print(game.get_pretty_board)
        self.assertFalse(game.isFinished())
