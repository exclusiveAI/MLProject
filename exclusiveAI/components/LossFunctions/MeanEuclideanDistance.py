import numpy as np
from .LossFunction import LossFunction

__all__ = ["MeanEuclideanDistance"]


class MeanEuclideanDistance(LossFunction):
    def __init__(self) -> None:
        super().__init__(
            name="Mean Euclidean Distance",
            function=lambda y_true, y_pred: np.mean(np.linalg.norm(y_true - y_pred, axis=-1)),
            derivative=lambda y_true, y_pred: (y_true - y_pred) / np.linalg.norm(y_true - y_pred, axis=-1).reshape(-1, 1)
        )