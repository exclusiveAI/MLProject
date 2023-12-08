__all__ = ['LossFunction']


class LossFunction:
    """
    Loss function class
    Args:

        name: Name of the loss function
        function: Loss function
        derivative: Derivative of the loss function
    """
    def __init__(self, name, function, derivative):
        self.name = name
        self.function = function
        self.function_derivative = derivative

    def __str__(self) -> str:
        return self.name
