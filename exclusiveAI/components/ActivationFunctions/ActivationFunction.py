__all__ = ["ActivationFunction"]


class ActivationFunction:
    """
    Class that represents an activation function.
    Attributes:
        name (str): The name of the activation function.
        function (function): The function itself.
        derivative (function): The activation function derivative.
    """
    def __init__(self, name, function, derivative):
        self.name = name
        self.function = function
        self.derivative = derivative

    def __str__(self) -> str:
        return self.name
