__all__ = ['Layer']

from exclusiveAI.components.ActivationFunctions import ActivationFunction
from exclusiveAI.components.Initializers import Initializer


class Layer:
    def __init__(self, units: int, inizializer: Initializer, activation_func: ActivationFunction, is_trainable=False):
        # Layers
        self.next: Layer = None
        self.prev: Layer = None

        self.output = None
        self.weights = None
        self.units = units
        self.is_trainable = is_trainable
        self.activation_func = activation_func
        self.initializer = inizializer
        self.is_initialized = False
        self.error = None
        self.nets = None

    def initialize(self, prev):
        self.weights = self.initializer.initialize(shape=(prev.units + 1, self.units))
        prev.next = self
        self.prev = prev
        self.is_initialized = True
        return self

    def feedforward(self, input):
        if not self.is_initialized:
            raise Exception("Layer not initialized")

        self.nets = self.weights @ input
        self.output = self.activation_func.function(self.nets)
        return self.output
