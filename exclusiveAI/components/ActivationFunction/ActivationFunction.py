__all__ = ["ActivationFunction"]


class ActivationFunction:
    def __init__(self, function, derivative) -> None:
        self.function = function
        self.derivative = derivative
