# This program is used to practice tic tac toe against itself. We use reenforced learning to train a neural network.

import os
import tensorflow as tf
import TicTacToe as ttt
import model_helper
import numpy
from copy import deepcopy
import config
import KIPlayer_2


def new_tensor_file_writer():
    index = 1
    tensor_board_log_dir = './tensorboard_log/better_train_data_' + repr(config.hidden_layers) + 'fullyLayers'\
                           + '_LR' + repr(config.learning_rate) \
                           + '_rewardDiscount' + repr(config.reward_discount) \
                           + '_boardSize' + repr(config.board_size) \
                           + '_config.runlength' + repr(config.runlength) \
                           + '_hidden_layer_size_factor' + repr(config.hidden_layer_size_factor) \
                           + '/'
    while os.path.isdir(tensor_board_log_dir + repr(index)):
        index += 1
    return tf.summary.FileWriter(tensor_board_log_dir + repr(index))

b = ttt.TicTacToe(config.board_size, config.board_size, config.runlength)

# Build model
x, y, pred, x_wins, o_wins, draw = model_helper.build_model(b.board_field_size, config.hidden_layers, config.hidden_layer_size_factor)

# Define loss and optimizer
cost = tf.reduce_mean(tf.squared_difference(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()


# get Q' with new state
# Q' are all the predicted possible rewards from the next state. Because the next move is made by the
# other player we have to go through all possible moves he makes and get our Q' from there.
def get_max_q_v1(game, player, sess):
    other_player = game.playerO if player == game.playerX else game.playerX
    Q1 = numpy.empty((0, 0))
    for m in range(game.board_width * game.board_height):
        m_idx_x = m % game.board_width
        m_idx_y = m // game.board_height
        if game.isFieldAvailable(m_idx_x, m_idx_y):
            game_copy = deepcopy(game)
            game_copy.setField(m_idx_x, m_idx_y, other_player)
            state1 = game_copy.board_for_learning_as_X() if player == game_copy.playerX else game_copy.board_for_learning_as_O()
            Q1 = numpy.append(Q1, sess.run(pred, feed_dict={x: [state1]}))
    # Use max of Q1. This is the maximum predicted Q-value of the action just made
    if Q1.size > 0:
        maxQ1 = numpy.max(Q1)
    else:
        maxQ1 = 0.0
    return maxQ1


def get_max_q_v2(game, player, sess):
    # predict next move of other player
    other_player = game.playerO if player == game.playerX else game.playerX
    idx_x, idx_y, _, _ = KIPlayer_2.next_move(game, sess, pred, x, state, 0.0)
    game_copy = deepcopy(game)
    game_copy.setField(idx_x, idx_y, other_player)
    state1 = game_copy.board_for_learning_as_X() if player == game_copy.playerX else game_copy.board_for_learning_as_O()
    Q1 = sess.run(pred, feed_dict={x: [state1]})
    maxQ1 = numpy.max(Q1)
    return maxQ1


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    writer = new_tensor_file_writer()
    writer.add_graph(sess.graph)
    tf.summary.scalar("cost", cost)
    summary_op = tf.summary.merge_all()

    for game_index in range(config.games_to_play):
        # Play a full game
        game = ttt.TicTacToe(config.board_size, config.board_size, config.runlength)
        players = [game.playerX, game.playerO]
        playerId = 0

        while not game.isFinished():
            player = players[playerId]

            # Get state
            if player == game.playerX:
                state = game.board_for_learning_as_X()
            else:
                state = game.board_for_learning_as_O()

            # Get next move
            current_exploration = config.exploration / config.games_to_play * (config.games_to_play - game_index) + 0.1
            idx_x, idx_y, allQ, action = KIPlayer_2.next_move(game, sess, pred, x, state, current_exploration)

            # set next move
            game.setField(idx_x, idx_y, player)

            maxQ1 = get_max_q_v1(game, player, sess)

            # find reward
            r = 0
            if game.isFinished():
                if game.isWinnerX() and player == game.playerX:
                    r = config.reward
                elif game.isWinnerO() and player == game.playerO:
                    r = config.reward

            # Update the target Q-Value for chosen action
            targetQ = allQ
            targetQ[action] = r + config.reward_discount * maxQ1

            # train network
            _, summary = sess.run([optimizer, summary_op], feed_dict={x: [state], y: [targetQ]})
            writer.add_summary(summary, game_index)

            # move to next player
            playerId = (playerId + 1) % 2

            # Print every one or other game
            if game_index % 500 == 0:
                print(game.get_pretty_board)

        if game_index % int(config.games_to_play / 100) == 0:
            print('finished', int(game_index / config.games_to_play * 100), '%')
