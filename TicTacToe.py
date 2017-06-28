import numpy


class TicTacToe:
    def __init__(self):
        self._empty = 0.0
        self._X = 1.0
        self._O = 2.0
        self._w, self._h = 3, 3
        self._board = [[self._empty for y in range(self._h)] for x in range(self._w)]

    @property
    def board_field_size(self):
        return numpy.size(self._board)

    @property
    def board_width(self):
        return self._w

    @property
    def board_height(self):
        return self._h

    @property
    def playerX(self):
        return self._X

    @property
    def playerO(self):
        return self._O

    @property
    def empty(self):
        return self._empty

    def board_for_learning(self):
        board = numpy.full((self._h * self._w), 0.0)
        for y in range(self._w):
            for x in range(self._h):
                mark = self._board[y][x]
                board[y * self._h + x] = mark
        return board


    def playX(self, x, y):
        self.setField(x, y, self._X)

    def playO(self, x, y):
        self.setField(x, y, self._O)

    def isFieldAvailable(self, x, y):
        return self._board[y][x] == self._empty

    def setField(self, x, y, player):
        value = self._board[y][x]
        if value == 0:
            self._board[y][x] = player

    def isWinner(self, player):
        horizontally = [player for x in range(self._w)]
        vertically = [player for y in range(self._h)]

        for x in range(self._w):
            row = self._board[x]
            if row == horizontally:
                return player

        # Rearrange board to get a column in an array
        row = [self._empty for y in range(self._w)]
        for x in range(self._w):
            for y in range(self._h):
                row[y] = self._board[y][x]
            if row == vertically:
                return player

        if self._board[0][0] == player and self._board[1][1] == player and self._board[2][2] == player:
            return player

        if self._board[0][2] == player and self._board[1][1] == player and self._board[2][0] == player:
            return player

    def isWinnerX(self):
        return self.isWinner(self._X)

    def isWinnerO(self):
        return self.isWinner(self._O)

    def isFinished(self):
        if self.isWinnerX():
            return True
        if self.isWinnerO():
            return True
        for row in range(self._h):
            for col in range(self._w):
                if self._board[row][col] == self._empty:
                    return False
        return True # Tie

    @property
    def get_pretty_board(self):
        b = ''.join('-' for x in range(self._w))
        for y in range(self._w):
            b += '\n'
            for x in range(self._w):
                b += str(self._board[y][x])
            b = b.replace(str(self._X), 'X')
            b = b.replace(str(self._O), 'O')
            b = b.replace(str(self._empty), ' ')
            b += '|'
        return b
