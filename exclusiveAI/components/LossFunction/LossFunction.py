__all__ = ['LossFunction']


class LossFunction:
    def __int__(self, function, derivative) -> None:
        self.loss_function = function
        self.loss_function_derivative = derivative
