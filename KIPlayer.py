import numpy
import random
from PIL import Image, ImageDraw, ImageFont


def next_move(game, sess, pred, x, exploration=0.0):
    board = game.board_for_learning()
    predictions = sess.run(pred, feed_dict={x: [board]})[0]
    idx_x = 0
    idx_y = 0

    # mix in some exploration
    if exploration > 0.0:
        for i in range(game.board_field_size):
            r = random.uniform(-exploration, exploration)
            predictions[i] = predictions[i] + r

    minimum = predictions.min()
    while True:
        i = numpy.argmax(predictions)
        idx_x = i % game.board_width
        idx_y = i // game.board_height
        if game.isFieldAvailable(idx_x, idx_y):
            break
        predictions[i] = minimum - 1.0
    return idx_x, idx_y


def visualize(game, sess, pred, x):
    board_for_learning = game.board_for_learning()
    predictions = sess.run(pred, feed_dict={x: [board_for_learning]})[0]
    predictions_for_display = numpy.reshape(predictions, (game.board_height, game.board_width))
    minimum = numpy.min(predictions)
    maximum = numpy.max(predictions)
    min_max_difference = maximum - minimum
    board_for_display = game._board

    img = Image.new("RGB", (game.board_width * 25, game.board_height * 25))
    drawing_context = ImageDraw.Draw(img)
    font = ImageFont.truetype("Arial.ttf", 15)

    for x in range(game.board_width):
        for y in range(game.board_height):
            pred_value = predictions_for_display[y][x]
            top_left = (x * 25, y * 25)
            bottom_right = ((x + 1) * 25, ((y + 1) * 25))
            color = int(255 / min_max_difference * (pred_value - minimum))
            drawing_context.rectangle([top_left, bottom_right], (255, color, color, 255))
            board_value = board_for_display[y][x]
            symbol = 'X' if board_value == game.playerX else 'O' if board_value == game.playerO else ' '
            drawing_context.text((25 * x + 7, 25 * y + 3), symbol, font=font, fill=(0, 0, 255, 255))

    return img