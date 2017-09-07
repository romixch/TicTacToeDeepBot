import numpy


class TicTacToe:

    def __init__(self, width = 3, height = 3, runlength = 3):
        self._empty = 0.0
        self._X = 1.0
        self._O = 2.0
        self._w, self._h = width, height
        self._runlength = runlength
        self._board = [[self._empty for y in range(self._h)] for x in range(self._w)]
        self._counter = 0

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
        return self.board_for_learning_as_X()


    def board_for_learning_as_X(self):
        board = numpy.full((self._h * self._w), 0.0)
        for y in range(self._w):
            for x in range(self._h):
                mark = self._board[y][x]
                board[y * self._h + x] = mark
        return board


    def board_for_learning_as_O(self):
        board = self.board_for_learning_as_X()
        board[board == self._X] = 5.0
        board[board == self._O] = self._X
        board[board == 5.0] = self._O
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
            self._counter += 1

    def isWinner(self, player):
        for x in range(self._w):
            row = self._board[x]
            if self._contains_winning_line(player, row):
                return player

        # Rearrange board to get a column in an array
        row = [self._empty for y in range(self._w)]
        for x in range(self._w):
            for y in range(self._h):
                row[y] = self._board[y][x]
            if self._contains_winning_line(player, row):
                return player

        # Diagonal top left
        for start_x in _inclusive_range(self._w - self._runlength):
            for start_y in _inclusive_range(self._w - self._runlength):
                row = [self._empty for y in range(self._runlength)]
                for r in range(self._runlength):
                    row[r] = self._board[start_y + r][start_x + r]
                if self._contains_winning_line(player, row):
                    return player

        # Diagonal top right
        for start_x in _inclusive_range(self._w - self._runlength):
            for start_y in _inclusive_range(self._h - self._runlength):
                row = [self._empty for y in range(self._runlength)]
                for r in range(self._runlength):
                    idy = start_y + r
                    idx = self._w - start_x - r - 1
                    row[r] = self._board[idy][idx]
                if self._contains_winning_line(player, row):
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
        b = ' ' + ''.join('-' for x in range(self._w))
        for y in range(self._w):
            b += '\n|'
            for x in range(self._w):
                b += str(self._board[y][x])
            b = b.replace(str(self._X), 'X')
            b = b.replace(str(self._O), 'O')
            b = b.replace(str(self._empty), ' ')
            b += '|'
        return b


    @property
    def next_turn(self):
        return self._X if self._counter % 2 == 0 else self._O

    def _contains_winning_line(self, player, row):
        strike = 0
        for i in range(len(row)):
            if row[i] == player:
                strike += 1
                if strike >= self._runlength: return True
            else:
                strike = 0
        return False

def _seq_in_seq(subseq, seq):
    while subseq[0] in seq:
        index = seq.index(subseq[0])
        if subseq == seq[index:index + len(subseq)]:
            return True
        else:
            seq = seq[index + 1:]
    else:
        return False

def _inclusive_range(end):
    return range(0, end + 1)