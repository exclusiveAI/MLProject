from .LossFunction import LossFunction
import mlx.core as mps

__all__ = ["CrossCorrelation"]


class CrossCorrelation(LossFunction):
    def __init__(self) -> None:
        super().__init__(
            name="Cross Correlation",
            function=self.function,
            derivative=self.derivative
        )

    @staticmethod
    def function(x, y):
        return -1 * mps.sum(x * y) / mps.sqrt(mps.sum(x * x) * mps.sum(y * y))

    @staticmethod
    def derivative(x, y):
        return (x * y) / mps.sqrt(mps.sum(x * x) * mps.sum(y * y))
