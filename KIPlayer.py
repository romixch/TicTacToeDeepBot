import numpy


def next_move(game, sess, pred, x):
    board = game.board_for_learning()
    predictions = sess.run(pred, feed_dict={x: [board]})[0]
    idx_x = 0
    idx_y = 0
    minimum = predictions.min()
    while True:
        i = numpy.argmax(predictions)
        idx_x = i % game.board_width
        idx_y = i // game.board_height
        if game.isFieldAvailable(idx_x, idx_y):
            break
        predictions[i] = minimum - 1.0
    return idx_x, idx_y