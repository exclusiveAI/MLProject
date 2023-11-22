__all__ = ['LossFunction']


class LossFunction:
    def __int__(self, name, function, derivative) -> None:
        self.name = name
        self.loss_function = function
        self.loss_function_derivative = derivative

    def __str__(self) -> str:
        return self.name
