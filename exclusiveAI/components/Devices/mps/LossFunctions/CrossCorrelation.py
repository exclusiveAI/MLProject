from .LossFunction import LossFunction
import mlx.core as mps

__all__ = ["CrossCorrelation"]


class CrossCorrelation(LossFunction):
    def __init__(self) -> None:
        super().__init__(
            name="Cross Correlation",
            function=lambda x, y: -1 * mps.sum(x * y) / mps.sqrt(mps.sum(x * x) * mps.sum(y * y)),
            derivative=lambda x, y: (x * y) / mps.sqrt(mps.sum(x * x) * mps.sum(y * y)),
        )
