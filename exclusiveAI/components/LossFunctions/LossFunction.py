__all__ = ['LossFunction']


class LossFunction:
    def __init__(self, name, function, derivative):
        self.name = name
        self.function = function
        self.function_derivative = derivative

    def __str__(self) -> str:
        return self.name
