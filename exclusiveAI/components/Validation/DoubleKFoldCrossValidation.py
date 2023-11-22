from .KFoldCrossValidation import KFoldCrossValidation
from ..neural_network import neural_network
import numpy as np

__all__ = ['DoubleKFoldCrossValidation']


class DoubleKFoldCrossValidation(KFoldCrossValidation):
    def __init__(self, outer_splits=5, inner_splits=5, shuffle=True, random_state=None):
        super.__init__(
            n_splits=outer_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        super().__init__(shuffle, random_state)
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def validate(self, model, x, y_true):
        outer_scores = []
        for outer_train_index, outer_val_index in self.split(x, y_true):
            x_train, x_val = np.take(x, outer_train_index), np.take(x, outer_val_index)
            y_train, y_val = np.take(y_true, outer_train_index), np.take(y_true, outer_val_index)

            inner_k_fold = KFoldCrossValidation(n_splits=self.inner_splits, shuffle=self.shuffle,
                                                random_state=self.random_state)
            inner_scores = []

            for inner_train_index, inner_val_index in inner_k_fold.split(x_train, y_train):
                x_inner_train, x_inner_val =  np.take(x_train, inner_train_index), np.take(x_train, inner_val_index)
                y_inner_train, y_inner_val = np.take(y_train, inner_train_index), np.take(y_train, inner_val_index)
                model.train(x_inner_train, y_inner_train)
                model.predict(x_inner_val)
                inner_score = model.layers[-1].error
                inner_scores.append(inner_score)

            average_inner_score = sum(inner_scores) / len(inner_scores)

            model.train(x_train, y_train)
            model.predict(x_val)
            outer_score = model.layers[-1].error
            outer_scores.append(outer_score)

        # select best
        best_model = np.argmin(outer_scores)
        model = neural_network(best_config)
        model.train(x, y_true)

        return model
