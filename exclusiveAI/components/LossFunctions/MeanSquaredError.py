import numpy as np
from .LossFunction import LossFunction

__all__ = ["MeanSquaredError"]


class MeanSquaredError(LossFunction):
    """
    Mean Squared Error loss function.
    """
    def __init__(self):
        super().__init__(
            name="Mean Squared Error",
            function=self.function,
            derivative=self.derivative
        )

    @staticmethod
    def function(y_true, y_pred):
        return np.mean(np.sum(np.square(y_true-y_pred), axis=1))

    @staticmethod
    def derivative(y_true, y_pred):
        return -(2 / y_true.shape[0]) * (y_true - y_pred)
