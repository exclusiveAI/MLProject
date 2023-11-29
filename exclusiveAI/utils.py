__all__ = ['utils', 'train_split', 'confusion_matrix', 'one_hot_encoding']

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


def confusion_matrix(predicted, target):
    """
    Compute the confusion matrix.
    """
    confusion = np.zeros((2, 2))
    for i in range(len(predicted)):
        j = round(predicted[i][0])
        k = round(target[i][0])
        confusion[j][k] += 1
    plt.matshow(confusion, cmap=plt.cm.Blues)
    # print num of element in the matrix
    for i in range(2):
        for j in range(2):
            plt.text(x=j, y=i, s=confusion[i][j], va='center', ha='center')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('res.png')


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
