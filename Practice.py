# This program is used to practice tic tac toe against itself. We use reenforced learning to train a neural network.

import os
import tensorflow as tf
import TicTacToe as ttt
import model_helper
import numpy
import KIPlayer
import pickle
import config


def new_tensor_file_writer():
    index = 1
    tensor_board_log_dir = './tensorboard_log/' + repr(config.hidden_layers) + 'fullyLayers'\
                           + '_LR' + repr(config.learning_rate) \
                           + '_rewardDiscount' + repr(config.reward_discount) \
                           + '_boardSize' + repr(config.board_size) \
                           + '_config.runlength' + repr(config.runlength) \
                           + '_hidden_layer_size_factor' + repr(config.hidden_layer_size_factor) \
                           + '/'
    while os.path.isdir(tensor_board_log_dir + repr(index)):
        index += 1
    return tf.summary.FileWriter(tensor_board_log_dir + repr(index))


def reshape(numpy_row_vector, num_columns):
    num_rows = int(len(numpy_row_vector) / num_columns)
    return numpy.reshape(numpy_row_vector, (num_rows, num_columns))


b = ttt.TicTacToe(config.board_size, config.board_size, config.runlength)


def calculate_discounted_rewards(states, actions, reward_number):
    rewards = numpy.empty((0, 0))
    for state_index in range(len(states)):
        distance_to_reward = len(states) - state_index - 1
        discounted_reward = reward_number * config.reward_discount ** distance_to_reward
        labels = numpy.full(b.board_field_size, 0.0)
        labels[int(actions[state_index])] = discounted_reward
        rewards = numpy.append(rewards, labels)
    rewards = reshape(rewards, b.board_field_size)
    return rewards


x, y, pred, x_wins, o_wins, draw = model_helper.build_model(b.board_field_size, config.hidden_layers, config.hidden_layer_size_factor)

# Define loss and optimizer
cost = tf.reduce_mean(tf.squared_difference(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sessX:
    with open('learn_x.dat', 'wb') as learn_x_file:
        with tf.Session() as sessO:
            with open('learn_o.dat', 'wb') as learn_o_file:
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

                for game_index in range(config.games_to_play):
                    # Play a full game
                    game = ttt.TicTacToe(config.board_size, config.board_size, config.runlength)
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
                            idx_x, idx_y = KIPlayer.next_move(game, sessX, pred, x, config.exploration)
                        else:
                            idx_x, idx_y = KIPlayer.next_move(game, sessO, pred, x, config.exploration)

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
                    is_winner_x = game.isWinnerX()
                    is_winner_o = game.isWinnerO()
                    winner = game.playerX if is_winner_x else game.playerO
                    rewards_winner = calculate_discounted_rewards(states[winner], actions[winner], config.reward)

                    looser = game.playerO if is_winner_x else game.playerX
                    punishments_looser = calculate_discounted_rewards(states[looser], actions[looser], config.punishment)

                    # Learn from past game as winner
                    x_wins_float = 1.0 if is_winner_x else 0.0
                    o_wins_float = 1.0 if is_winner_o else 0.0
                    draw_float = 1.0 if not is_winner_x and not is_winner_o else 0.0
                    sess_winner = sessX if is_winner_x else sessO
                    _, summary = sess_winner.run([optimizer, summary_op], feed_dict={x: states[winner], y: rewards_winner, x_wins: x_wins_float, o_wins: o_wins_float, draw: draw_float})
                    writer.add_summary(summary, game_index)

                    # Learn from past game as looser
                    x_wins_array = 1.0 if is_winner_x else 0.0
                    o_wins_float = 1.0 if is_winner_o else 0.0
                    draw_float = 1.0 if not is_winner_x and not is_winner_o else 0.0
                    sess_looser = sessO if is_winner_x else sessX
                    _, summary = sess_looser.run([optimizer, summary_op], feed_dict={x: states[looser], y: punishments_looser, x_wins: x_wins_float, o_wins: o_wins_float, draw: draw_float})
                    writer.add_summary(summary, game_index)

                    pickle.dump(states[game.playerX], learn_x_file)
                    pickle.dump(actions[game.playerX], learn_x_file)
                    pickle.dump(states[game.playerO], learn_o_file)
                    pickle.dump(actions[game.playerO], learn_o_file)
                    pickle.dump(rewards_winner, learn_x_file) if is_winner_x else pickle.dump(punishments_looser, learn_o_file)
                    pickle.dump(punishments_looser, learn_o_file) if is_winner_x else pickle.dump(punishments_looser, learn_x_file)

                    if game_index % int(config.games_to_play / 100) == 0:
                        print('finished', int(game_index / config.games_to_play * 100), '%')

                learn_o_file.close()
            model_helper.save_model(sessO, 'tf-model/O', x, pred)
            learn_x_file.close()
    model_helper.save_model(sessX, 'tf-model/X', x, pred)