import TicTacToe as t
import random
import pickle
import KI

def play_games(boards_filename, labels_filename, games):
    ki = KI.TicTacToeKI(3 * 3 * 2, 3 * 3)
    boards_file = open(boards_filename, 'w')
    labels_file = open(labels_filename, 'w')
    print('Playing {0:d} games now:'.format(games))
    for game in range(0, games):
        ttt = t.TicTacToe()
        players = [ttt.playerX, ttt.playerO]
        playerId = 0
        boards = {ttt.playerX: [], ttt.playerO: []}
        labels = {ttt.playerX: [], ttt.playerO: []}

        played_moves = 0
        while not ttt.isFinished():
            player = players[playerId]
            x, y = ki.play_move(ttt, player)
            boards[player].append(ttt.board_for_learning(player))
            label = [0.0 for i in range(9)]
            label[x + y * 3] = 1.0
            labels[player].append(label)
            ttt.setField(x, y, player)
            played_moves += 1
            playerId = (playerId + 1) % 2

        if played_moves < 6:
            if ttt.isWinnerX():
                pickle.dump(boards[ttt.playerX], boards_file)
                pickle.dump(labels[ttt.playerX], labels_file)
            if ttt.isWinnerO():
                pickle.dump(boards[ttt.playerO], boards_file)
                pickle.dump(labels[ttt.playerO], labels_file)

        if game % 1000 == 0:
            print('Played {0:d} games...'.format(game))

    boards_file.close()
    labels_file.close()
    print('Finished playing')

print('Generating traning set:')
play_games('boards_train.p', 'labels_train.p', 10000)

print('Generating test set:')
play_games('boards_test.p', 'labels_test.p', 1000)
