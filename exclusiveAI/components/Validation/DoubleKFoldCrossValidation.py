from .KFoldCrossValidation import KFoldCrossValidation
from exclusiveAI.ConfiguratorGen import ConfiguratorGen
from ..neural_network import neural_network
import numpy as np

__all__ = ['DoubleKFoldCrossValidation']


class DoubleKFoldCrossValidation:
    def __init__(self, n_splits=5, outer_splits=5, inner_splits=5, shuffle=True, random_state=None,
                 configurator: ConfiguratorGen = None):
        self.n_splits = n_splits
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.configurator = configurator

    def split(self, x, y_true=None):
        if y_true is None and self.shuffle:
            raise ValueError("y must be specified if shuffle is True")

        indices = np.arange(len(x))
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(indices)

        fold_sizes = np.full(self.n_splits, len(x) // self.n_splits, dtype=int)
        fold_sizes[:len(x) % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield list(indices[np.concatenate((indices[:start], indices[stop:]))]), list(indices[start:stop])
            current = stop

    def validate(self, x, y_true):
        outer_scores = []
        for model, config in self.configurator:
            for outer_train_index, outer_val_index in self.split(x, y_true):
                x_train, x_val = np.take(x, outer_train_index), np.take(x, outer_val_index)
                y_train, y_val = np.take(y_true, outer_train_index), np.take(y_true, outer_val_index)

                inner_k_fold = KFoldCrossValidation(n_splits=self.inner_splits, shuffle=False,
                                                    random_state=self.random_state)
                inner_best_model = None

                for inner_train_index, inner_val_index in inner_k_fold.split(x_train, y_train):
                    x_inner_train, x_inner_val = np.take(x_train, inner_train_index), np.take(x_train, inner_val_index)
                    y_inner_train, y_inner_val = np.take(y_train, inner_train_index), np.take(y_train, inner_val_index)
                    model.train(x_inner_train, y_inner_train, x_inner_val, y_inner_val)
                    if inner_best_model is None or model.get_last()['val_mse'] < inner_best_model.get_last()['val_mse']:
                        inner_best_model = model

                outer_scores.append(model.evaluate(x_val, y_val))

        return np.mean(outer_scores)
