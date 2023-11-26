import numpy as np
from .LossFunction import LossFunction

__all__ = ["MeanSquaredError"]


class MeanSquaredError(LossFunction):
    def __init__(self):
        super().__init__(
            name="Mean Squared Error",
            function=lambda y_true, y_pred: np.mean(np.sum(np.square(np.subtract(y_true, y_pred)), axis=1)),
            derivative=lambda y_true, y_pred: (-(2 / y_true.shape[0]) * np.subtract(y_true, y_pred))
        )
