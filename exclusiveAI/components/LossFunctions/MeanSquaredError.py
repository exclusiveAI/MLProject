import numpy as np
from numpy import ndarray

from .LossFunction import LossFunction

__all__ = ["MeanSquaredError"]


class MeanSquaredError(LossFunction):
    def __init__(self):
        super().__init__(
            name="Mean Squared Error",
            loss_function=lambda y_true, y_pred: np.mean(np.sum(np.square(np.subtract(y_true, y_pred)), axis=1)),
            loss_function_derivative=lambda y_true, y_pred: (-2 * np.subtract(y_true, y_pred)) * np.subtract(y_true,
                                                                                                             y_pred)
        )
