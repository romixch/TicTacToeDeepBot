import KIPlayer
import TicTacToe as ttt
import model_helper
import tensorflow as tf


continue_playing = True

with tf.Session() as sess:
    x_tensor, pred_tensor = model_helper.load_model(sess, 'tf-model/X')
    while continue_playing:
        game = ttt.TicTacToe()
        while not game.isFinished():
            x, y = KIPlayer.next_move(game, sess, pred_tensor, x_tensor)
            game.setField(x, y, game.playerX)
            print(game.get_pretty_board)

            if (game.isFinished()):
                if game.isWinnerX():
                    print('You have lost!')
                else:
                    print('You played a draw.')
                break

            while not game.isFieldAvailable(x, y):
                x = int(input('x: '))
                y = int(input('y: '))
            game.setField(x, y, game.playerO)

            if game.isFinished():
                if game.isWinnerO():
                    print('You won!!!')
                else:
                    print('You played a draw.')
                break
        continue_playing = input('Start over? (y / n)') == 'y'
