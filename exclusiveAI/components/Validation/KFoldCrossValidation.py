import numpy as np

from ..NeuralNetwork import NeuralNetwork
from exclusiveAI.ConfiguratorGen import ConfiguratorGen

__all__ = ["KFoldCrossValidation"]


class KFoldCrossValidation:
    """
    K-fold cross validation.

    Args:
        n_splits (int): Number of splits
        shuffle (bool): true if you want to shuffle the data
        random_state (int): random seed
        configurator (ConfiguratorGen): a particular model configuration (i.e. hyperparameters config.)
    Attributes:
        n_splits (int): Number of splits
        shuffle (bool): true if you want to shuffle the data
        random_state (int): random seed
        configurator (ConfiguratorGen): a particular model configuration (i.e. hyperparameters config.)
        best_config (dict): the best configuration found
        best_model (NeuralNetwork): the best model found
    """
    def __init__(self, n_splits=5, shuffle=True, random_state=None, configurator: ConfiguratorGen = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.configurator = configurator
        self.best_config = None
        self.best_model = None

    def split(self, x, y_true=None):
        """
        Data splitting for cross validation
        Args:
            x: input
            y_true: target

        Raises:
            ValueError: if y_true is not specified and shuffle is True.

        Returns: a tuple containing two lists of indices:
                - The training indices obtained by excluding the validation part in the fold.
                - The validation indices for the current fold.
        """
        if y_true is None and self.shuffle:
            raise ValueError("y must be specified if shuffle is True")

        indices = np.arange(len(x)) #create an array of indices
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(indices)

        # Calculate the base size of each fold.
        fold_sizes = np.full(self.n_splits, len(x) // self.n_splits, dtype=int)
        # The remainder of this division is distributed to ensure that all samples are included.
        fold_sizes[:len(x) % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield list(indices[np.concatenate((indices[:start], indices[stop:]))]), list(indices[start:stop])
            current = stop

    def validate(self, x, y_true):
        """
        Perform a k-fold cross validation.

        Args:
            x: the input
            y_true: the target

        Returns: the best configuration found

        """
        for model, config in self.configurator:
            for train_index, val_index in zip(self.split(x, y_true)):
                x_train, x_val = np.take(x, train_index), np.take(x, val_index)
                y_train, y_val = np.take(y_true, train_index), np.take(y_true, val_index)

                model.train(x_train, y_train, x_val, y_val)
                score = model.get_last()['val_mse']
                if self.best_model is None or score < self.best_model.get_last()['val_mse']:
                    self.best_model = model
                    self.best_config = config

        return self.best_config
