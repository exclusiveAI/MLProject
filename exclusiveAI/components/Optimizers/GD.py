from .Optimizer import Optimizer

class GD(Optimizer):
    def __init__(self):
        self.__init__()

    #     Gradient Descent algorithm
    def apply(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.gradients[i]
        return self.weights
