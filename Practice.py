# This program is used to practice tic tac toe against itself. We use reenforced learning to train a neural network.

import os
import tensorflow as tf
import TicTacToe as ttt
import model_helper
import numpy
import KIPlayer

# Parameters
learning_rate = 1e-4
hidden_layers = 2
games_to_play = 1000000
reward_discount = 0.7
punishment_discount = 0.7
reward = 1.0
punishment = -1.0


def new_tensor_file_writer():
    index = 1
    tensor_board_log_dir = './tensorboard_log/' + repr(hidden_layers) + 'fullyLayers'\
                           + '_LR' + repr(learning_rate) \
                           + '_rewardDiscount' + repr(reward_discount) \
                           + '/'
    while os.path.isdir(tensor_board_log_dir + repr(index)):
        index += 1
    return tf.summary.FileWriter(tensor_board_log_dir + repr(index))


def reshape(numpy_row_vector, num_columns):
    num_rows = int(len(numpy_row_vector) / num_columns)
    return numpy.reshape(numpy_row_vector, (num_rows, num_columns))

b = ttt.TicTacToe()

def calculate_discounted_rewards(states, actions, reward_number):
    rewards = numpy.empty((0, 0))
    for state_index in range(len(states)):
        distance_to_reward = len(states) - state_index - 1
        discounted_reward = reward_number * reward_discount ** distance_to_reward
        labels = numpy.full(b.board_field_size, 0.0)
        labels[int(actions[state_index])] = discounted_reward
        rewards = numpy.append(rewards, labels)
    rewards = reshape(rewards, b.board_field_size)
    return rewards


x, y, pred, x_wins, o_wins, draw = model_helper.build_model(b.board_field_size, hidden_layers)

# Define loss and optimizer
cost = tf.reduce_mean(tf.squared_difference(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sessX:
    with tf.Session() as sessO:
        sessX.run(init)
        sessO.run(init)
        writer = new_tensor_file_writer()
        writer.add_graph(sessX.graph)
        writer.add_graph(sessO.graph)
        tf.summary.scalar("cost", cost)
        tf.summary.scalar('x_wins', x_wins)
        tf.summary.scalar('o_wins', o_wins)
        tf.summary.scalar('draw', draw)
        summary_op = tf.summary.merge_all()

        for game_index in range(games_to_play):
            # Play a full game
            game = ttt.TicTacToe()
            players = [game.playerX, game.playerO]
            playerId = 0
            states = {
                game.playerX: numpy.empty((0, 0)),
                game.playerO: numpy.empty((0, 0))
            }
            actions = {
                game.playerX: [],
                game.playerO: []
            }
            while not game.isFinished():
                player = players[playerId]

                # Record board state
                states[player] = numpy.append(states[player], game.board_for_learning())

                # Calculate next move
                if player == game.playerX:
                    idx_x, idx_y = KIPlayer.next_move(game, sessX, pred, x)
                else:
                    idx_x, idx_y = KIPlayer.next_move(game, sessO, pred, x)

                # set next move
                game.setField(idx_x, idx_y, player)

                # record played move
                field = idx_y * game.board_width + idx_x
                actions[player] = numpy.append(actions[player], field)

                # move to next player
                playerId = (playerId + 1) % 2

                # Print every one or other game
                if game_index % 500 == 0:
                    print(game.get_pretty_board)

            # reshape numpy arrays
            states[game.playerX] = reshape(states[game.playerX], game.board_field_size)
            states[game.playerO] = reshape(states[game.playerO], game.board_field_size)

            # Calculate discounted rewards for learning
            winner = game.playerX if game.isWinnerX() else game.playerO
            rewards_winner = calculate_discounted_rewards(states[winner], actions[winner], reward)

            looser = game.playerO if game.isWinnerX() else game.playerX
            punishments_looser = calculate_discounted_rewards(states[looser], actions[looser], punishment)

            # Learn from past game as winner
            x_wins_float = 1.0 if game.isWinnerX() else 0.0
            o_wins_float = 1.0 if game.isWinnerO() else 0.0
            draw_float = 1.0 if not game.isWinnerX() and not game.isWinnerO() else 0.0
            sess_winner = sessX if game.isWinnerX() else sessO
            _, summary = sess_winner.run([optimizer, summary_op], feed_dict={x: states[winner], y: rewards_winner, x_wins: x_wins_float, o_wins: o_wins_float, draw: draw_float})
            writer.add_summary(summary, game_index)

            # Learn from past game as looser
            x_wins_array = 1.0 if game.isWinnerX() else 0.0
            o_wins_float = 1.0 if game.isWinnerO() else 0.0
            draw_float = 1.0 if not game.isWinnerX() and not game.isWinnerO() else 0.0
            sess_looser = sessO if game.isWinnerX() else sessX
            _, summary = sess_looser.run([optimizer, summary_op], feed_dict={x: states[looser], y: punishments_looser, x_wins: x_wins_float, o_wins: o_wins_float, draw: draw_float})
            writer.add_summary(summary, game_index)

            if game_index % int(games_to_play / 100) == 0:
                print('finished', int(game_index / games_to_play * 100), '%')

        model_helper.save_model(sessO, 'tf-model/O', x, pred)
    model_helper.save_model(sessX, 'tf-model/X', x, pred)