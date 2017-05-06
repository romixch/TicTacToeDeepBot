# This is currently a dataset for tic tac toe games with a matrix of 3 by 3.

import numpy
import pickle


class Data():
    def __init__(self):
        self._train = DataSet('boards_train.p', 'labels_train.p')
        self._test = DataSet('boards_test.p', 'labels_test.p')

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
    def __init__(self, boards_filename, labels_filename):
        self._boards = []
        self._batch_pos = 0

        boards_file = open(boards_filename, 'rb')
        while True:
            try:
                boards = pickle.load(boards_file)
                for b in boards:
                    if numpy.size(self._boards) == 0:
                        self._boards = b
                    else:
                        self._boards = numpy.vstack([self._boards, b])
            except EOFError:
                break
        boards_file.close()

        self._labels = []
        labels_file = open(labels_filename, 'rb')
        while True:
            try:
                labels = pickle.load(labels_file)
                for l in labels:
                    if numpy.size(self._labels) == 0:
                        self._labels = l
                    else:
                        self._labels = numpy.vstack([self._labels, l])
            except EOFError:
                break
        labels_file.close()

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
