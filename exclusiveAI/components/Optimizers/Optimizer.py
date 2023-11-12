__all__ = ['Optimizer']


class Optimizer:
    def __init__(self, learning_rate: float, regularization: float, momentum: float):
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.momentum = momentum
        self.old_deltas = []
