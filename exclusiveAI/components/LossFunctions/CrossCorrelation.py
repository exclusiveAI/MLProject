from .LossFunction import LossFunction
import numpy as np

__all__ = ["CrossCorrelation"]


class CrossCorrelation(LossFunction):
    def __init__(self) -> None:
        super().__init__(
            name="Cross Correlation",
            function=self.function,
            derivative=self.derivative,
        )

    @staticmethod
    def function(x: np.ndarray, y: np.ndarray):
        return -1 * np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y), dtype=np.float64)

    @staticmethod
    def derivative(x: np.ndarray, y: np.ndarray):
        return (x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y), dtype=np.float64)