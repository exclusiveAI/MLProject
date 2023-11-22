__all__ = ["ActivationFunction"]


class ActivationFunction:
    def __init__(self, name, function, derivative):
        self.name = name
        self.function = function
        self.derivative = derivative

    def __str__(self) -> str:
        return self.name
