# TicTacToeDeepBot

Our litle machine learning experiment.

## What is it?

TicTacToeDeepBot is training a neural network to play Tic Tac Toe using tensorflow.
After succeeding in this challange we are aiming to solve some more difficult problems like "Connect Four".

## How to use

There are two main functions:

- Practice.py: This program plays games against itself and trains a neural network with a reinforcement learning algorythm.
- PlayInteractively.py: Lets you use our lately trained network to play against it interactively.

## TODO

We think of the following next steps:

- Connect to https://github.com/K2InformaticsGmbH/egambo and take the challange

## Thoughts

This are thoughts that may help creating a better model. But we are not sure wheather the
advantage is really significant or not.

## Interesting details

Currently we train two models: One for the player X and one for O. This is because in Tic Tac Toe it is a difference if
we play as first mover or not. In bigger boards this may be not interesting and necessary anymore.

You can use tensorboard to monitor your network while it is trained. We log the cost function and other parameters to
`./tensorboard_log`,

# Setup the project

We recommend to use a virtual environment for python3. There are severaltutorials on how to create this environment.
It may look like this:

```
# Somehow setup the environment
<TODO: add code>
# Get 
source ./virtenv/bin/activate
```

## Technologies

This Project uses Tensorflow. It would be interesting to make the same with following other technology candidates:

- Pytorch
- Keras
- Deeplearning4j
