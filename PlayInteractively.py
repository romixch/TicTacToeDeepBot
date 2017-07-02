import KIPlayer
import HumanPlayer
import TicTacToe as ttt
import model_helper
import tensorflow as tf
import random


continue_playing = True

print('loading AI...')
with tf.Session() as sessX:
    x_tensor, pred_tensor = model_helper.load_model(sessX, 'tf-model/X')
    with tf.Session() as sessO:
        model_helper.load_model(sessO, 'tf-model/O')
        while continue_playing:
            game = ttt.TicTacToe()
            # choose starter
            if random.choice([True, False]):
                fun_to_play_X = KIPlayer.next_move
                fun_to_play_O = HumanPlayer.next_move
                print('X: Computer, O: You')
            else:
                fun_to_play_X = HumanPlayer.next_move
                fun_to_play_O = KIPlayer.next_move
                print('X: You, O: Computer')

            while not game.isFinished():
                if game.next_turn == game.playerO:
                    x, y = fun_to_play_O(game, sessO, pred_tensor, x_tensor)
                    game.setField(x, y, game.playerO)
                else:
                    x, y = fun_to_play_X(game, sessX, pred_tensor, x_tensor)
                    game.setField(x, y, game.playerX)

                print(game.get_pretty_board)

                if game.isFinished():
                    if game.isWinnerX():
                        print('X has won!')
                    elif game.isWinnerO():
                        print('O has won!')
                    else:
                        print('You played a draw.')
                    break

            continue_playing = input('Start over? (y / n)') == 'y'
