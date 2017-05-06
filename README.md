# TicTacToeDeepBot

Our litle machine learning experiment.

## What is it?

TicTacToeDeepBot is training a neural network to play Tic Tac Toe using tensorflow.
After succeeding in this challange we are aiming to solve some more difficult problems like "Connect Four".

## How to use

There are two main functions:

- Play.py: This is playing many many games against itself and collects the best games as training data.
- Train.py: This trains the neural network to learn from the best games played with Play.py.

In order to get a good model you can alternately play and train. As with Tic Tac Toe it will soon converge.

## TODO

We think of the following next steps:

- Create a little program to play against the bot interactively to actually see what the model does.
- Connect to https://github.com/K2InformaticsGmbH/egambo and take the challange

## Thoughts

This are thoughts that may help creating a better model. But we are not sure wheather the
advantage is really significant or not.

### First mover or not
In small boards like Tic Tac Toe (3 x 3) it is a quite different thing if you have the first move or not. I would like 
to take this into consideration. There would be several ways to do this:

- Maybe feed it into the model as an extra input parameter.
- Consider it at collecting data as a draw can be considered a good game as the second mover.
- Create a completely new model for those two players.

### Rotated boards
As boards are growing there will be a considerably big amount of possible states:

- 3 x 3: 362880
- 4 x 4: 20922789888000
- 5 x 5: 15511210043330985984000000

It might be advantageous to reduce the amount of states by just rotating it. Because all of this boards are actually the same:

|Board | 90 °| 180 ° | 270 °|
|------|-----|-------|------|
|`X O` |`  X`|`   `  |`O  ` |
|`   ` |`   `|`   `  |`   ` |
|`   ` |`  O`|`O X`  |`X  ` |

