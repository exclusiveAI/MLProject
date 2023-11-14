__all__ = ['Optimizer']


class Optimizer:
    def __init__(self, learning_rate: float, regularization: float):
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.old_deltas = []
