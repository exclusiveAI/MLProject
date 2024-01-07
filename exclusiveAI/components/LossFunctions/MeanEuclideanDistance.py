import numpy as np
from .LossFunction import LossFunction

__all__ = ["MeanEuclideanDistance"]


class MeanEuclideanDistance(LossFunction):
    """
    Mean Euclidean Distance loss fucntion.
    """
    def __init__(self) -> None:
        super().__init__(
            name="Mean Euclidean Distance",
            function=self.function,
            derivative=self.derivative
        )

    @staticmethod
    def function(y_true, y_pred):
        return np.mean(np.sqrt(np.sum((np.square(y_true-y_pred, dtype='float32')), axis=1, dtype='float32'), dtype='float32'), dtype='float32')

    @staticmethod
    def derivative(y_true, y_pred):
        return -1 / y_pred.shape[0] * (y_true-y_pred) / np.linalg.norm(y_true - y_pred, axis=1).reshape(-1, 1)
