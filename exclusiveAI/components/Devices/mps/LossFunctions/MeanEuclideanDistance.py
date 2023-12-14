import mlx.core as mps
from .LossFunction import LossFunction

__all__ = ["MeanEuclideanDistance"]


class MeanEuclideanDistance(LossFunction):
    """
    Mean Euclidean Distance loss function.
    """
    def __init__(self) -> None:
        super().__init__(
            name="Mean Euclidean Distance",
            function=self.function,
            derivative=self.derivative
        )

    @staticmethod
    def function(y_true, y_pred):
        return mps.mean(mps.sqrt(mps.sum(mps.square(y_true-y_pred), axis=1)))

    @staticmethod
    def derivative(y_true, y_pred):
        return -1 / y_pred.shape[0] * (y_true-y_pred) / mps.sqrt(mps.sum((mps.square(y_true-y_pred)), axis=1)).reshape(-1, 1)
