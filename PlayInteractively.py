import KI
import TicTacToe as t

ki = KI.TicTacToeKI(3 * 3 * 2, 3 * 3)
ttt = t.TicTacToe()

while not ttt.isFinished():
    x, y = ki.play_move(ttt, ttt.playerX)
    ttt.setField(x, y, ttt.playerX)
    print(ttt.get_pretty_board)

    if (ttt.isFinished()):
        if ttt.isWinnerX():
            print('You have lost!')
        else:
            print('You played a draw.')
        break

    while not ttt.isFieldAvailable(x, y):
        x = int(input('x: '))
        y = int(input('y: '))
    ttt.setField(x, y, ttt.playerO)

    if (ttt.isFinished()):
        if ttt.isWinnerO():
            print('You won!!!')
        else:
            print('You played a draw.')
        break
