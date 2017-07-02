
def next_move(game, sess, pred_tensor, x_tensor):
    legal_choice = False
    while not legal_choice:
        x = int(input('x: '))
        y = int(input('y: '))
        legal_choice = game.isFieldAvailable(x, y)
    return x, y