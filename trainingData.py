# This is currently a dataset for tic tac toe games with a matrix of 3 by 3.

import numpy
import pickle


class Data():
    def __init__(self):
        self._train = DataSet('training.csv', 16, 16)
        self._test = DataSet('test.csv', 16, 16)

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test

    @property
    def number_of_features(self):
        return numpy.size(self._train.boards[0])

    @property
    def number_of_output_classes(self):
        return numpy.size(self._train.labels[0])


class DataSet:
    def __init__(self, file_name, state_length,  number_of_labels):
        self._boards = numpy.full(state_length, 32.0)
        self._labels = numpy.full(number_of_labels, 0.5)
        self._batch_pos = 0

        with open(file_name, 'r') as file:
            file.readline()

            lines = file.read().split("\n")

            for line in lines:
                values = line.replace('"', '').split(';')
                state = [ord(value) for value in values[0]]
                winner = values[4]
                reward = 1.0 if winner == 'X' else -1.0
                discounted_reward = reward * self.Y ** int(values[3])
                field = int(values[2])
                labels = numpy.full(number_of_labels, 0.0)
                labels[field] = discounted_reward

                self._boards = numpy.vstack([self._boards, state])
                self._labels = numpy.vstack([self._labels, labels])


    @property
    def Y(self):
        return 0.6

    @property
    def num_examples(self):
        return self._boards.shape[0]

    @property
    def boards(self):
        return self._boards

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size, fake_data=False):
        board_batch = self._boards[self._batch_pos : self._batch_pos + batch_size]
        label_batch = self._labels[self._batch_pos: self._batch_pos + batch_size]
        self._batch_pos = self._batch_pos + batch_size
        return board_batch, label_batch

    def reset_batch(self):
        self._batch_pos = 0
