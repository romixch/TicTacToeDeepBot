import TicTacToe as t
import pickle
import KI

def play_games(boards_filename, labels_filename, games_to_play):
    ki = KI.TicTacToeKI(3 * 3 * 2, 3 * 3, randomness=2)
    boards_file = open(boards_filename, 'wb')
    labels_file = open(labels_filename, 'wb')
    print('Playing {0:d} games now:'.format(games_to_play))
    collected_games = 0
    while collected_games < games_to_play:
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
            if collected_games % 1000 == 0:
                print(ttt.get_pretty_board)

        if played_moves < 10:
            collected_games += 1
            if ttt.isWinnerX():
                pickle.dump(boards[ttt.playerX], boards_file)
                pickle.dump(labels[ttt.playerX], labels_file)
            if ttt.isWinnerO():
                pickle.dump(boards[ttt.playerO], boards_file)
                pickle.dump(labels[ttt.playerO], labels_file)

            if collected_games % 1000 == 0:
                print('Collected {0:d} games...'.format(collected_games))

    boards_file.close()
    labels_file.close()
    print('Finished playing')

print('Generating traning set:')
play_games('boards_train.p', 'labels_train.p', 10000)

print('Generating test set:')
play_games('boards_test.p', 'labels_test.p', 1000)
