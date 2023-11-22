import numpy as np

from ..neural_network import neural_network

__all__ = ["KFoldCrossValidation"]


class KFoldCrossValidation:
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

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

    def validate(self, x, y_true, configs):
        scores = []
        for train_index, val_index, config in zip(self.split(x, y_true), configs):
            model = neural_network(config)
            x_train, x_val = np.take(x, train_index), np.take(x, val_index)
            y_train, y_val = np.take(y_true, train_index), np.take(y_true, val_index)

            model.train(x_train, y_train)
            score = model.layers[-1].error
            scores.append(score)

        # select best model
        best_model = np.argmin(scores)
        best_config = configs[best_model]
        model = neural_network(best_config)
        model.train(x, y_true)

        return model
