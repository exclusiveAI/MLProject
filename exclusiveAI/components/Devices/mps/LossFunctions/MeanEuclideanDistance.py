import mlx.core as mps
from .LossFunction import LossFunction

__all__ = ["MeanEuclideanDistance"]


class MeanEuclideanDistance(LossFunction):
    """
    Mean Euclidean Distance loss fucntion.
    """
    def __init__(self) -> None:
        super().__init__(
            name="Mean Euclidean Distance",
            function=lambda y_true, y_pred: mps.mean(mps.sqrt(mps.sum(mps.square(y_true-y_pred), axis=1))),
            derivative=lambda y_true, y_pred: -1 / y_pred.shape[0] * (y_true-y_pred) / mps.sqrt(mps.sum((mps.square(y_true-y_pred)), axis=1)).reshape(-1, 1)
        )