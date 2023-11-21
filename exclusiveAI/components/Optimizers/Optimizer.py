__all__ = ['Optimizer']
from exclusiveAI.components.ActivationFunctions import ActivationFunction
from exclusiveAI.components.Layers import Layer
class Optimizer:
    def __init__(self, learning_rate: float, regularization: float, activation_func: ActivationFunction, layer: Layer):
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.activation_function = activation_func
        self.layer = layer
        self.old_dw = []

    def calulate_deltas(self, model, y_true, x):
        model.predict(x)
        
        layers = model.layers
        
        deltas = []
        for layer in layers.reverse():
            # Pass y_true
            deltas.insert(0, layer.backpropagate(y_true))
        return deltas      