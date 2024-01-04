__all__ = ['utils', 'train_split', 'one_hot_encoding', 'plot_history']

import numpy as np
import matplotlib.pyplot as plt


def split_batches(inputs, input_label, batch_size):
    """
    Split the inputs into batches of size batch_size.
    """
    for i in range(0, len(inputs), batch_size):
        # if last
        if i + batch_size > len(inputs):
            batch_size = len(inputs) - i
        yield inputs[i:i + batch_size], input_label[i:i + batch_size]


def myzip(*iterables):
    sentinel = object()
    iterators = [iter(it) for it in iterables if it]
    while iterators:
        result = []
        for it in iterators:
            elem = next(it, sentinel)
            if elem is sentinel:
                return
            result.append(elem)
        yield tuple(result)


def train_split(inputs, input_label, split_size=0.2, shuffle=True, random_state=42):
    """
    Split the data into training and test sets.
    """
    indices = np.arange(len(inputs))
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(indices)
    # Left size is training set, 1- split size * the # of the inputs
    left_indices = indices[:int((1 - split_size) * len(indices))]
    # Right size is the test/validation set, inputs - training set
    right_indices = indices[len(left_indices):]
    return inputs[left_indices], input_label[left_indices], inputs[right_indices], input_label[
        right_indices], left_indices, right_indices


def plot_history(lines: dict, fig_size=(10, 6)):
    plt.figure(figsize=fig_size)
    for elem in lines:
        plt.plot(lines[elem], label=elem)
    plt.legend()
    plt.show()


def one_hot_encoding(y):
    """
    One hot encoding.
    """
    for column in y.T:
        col_max = column.max()
        new_col = np.zeros((len(column), col_max))
        for i in range(len(column)):
            new_col[i][column[i] - 1] = 1
        column = new_col
        y = np.hstack((y, column))
        y = y[:, 1:]
    return y


