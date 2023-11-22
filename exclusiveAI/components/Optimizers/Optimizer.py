__all__ = ['Optimizer']
from exclusiveAI.components.Schedulers import LearningRateScheduler
from exclusiveAI.components.ActivationFunctions import ActivationFunction
class Optimizer:
    # def __init__(self, learning_rate: float, regularization: float, activation_func: ActivationFunction, lr_scheduler: LearningRateScheduler):
    def __init__(self, learning_rate: float, regularization: float, activation_func: ActivationFunction,
                     lr_scheduler= None):
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.activation_function = activation_func
        self.lr_scheduler = lr_scheduler
        self.old_dw = []

    def calulate_deltas(self, model, y_true, x):
        model.predict(x)
        
        layers = list(model.layers)
        
        deltas = []
        for layer in reversed(layers):
            # Pass y_true
            deltas.insert(0, layer.backpropagate(y_true))
        return deltas      
    
    def update_lr(self):
        self.learning_rate = self.lr_scheduler.update(self.learning_rate)

    def update(self, model, y_true, x):
        pass