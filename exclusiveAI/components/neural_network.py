import numpy as np
from exclusiveAI.components import Layers

from exclusiveAI.components.Layers import InputLayer, Layer
from exclusiveAI.components.LossFunctions import LossFunction
from exclusiveAI.components.Optimizers import Optimizer


class neural_network:
    def __init__(self, layers: list, loss: LossFunction, optimizer: Optimizer, learning_rate: float, epochs: int, callbacks: list):
        self.learning_rate = learning_rate
        
        self.layers = layers
        self.loss_function = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.callbacks = callbacks
        
        self.history = []
        
        self.initialize()
        
    def initialize(self):
        self.layers[0].initialize()
        for i, layer in enumerate(self.layers[1:]):
            layer.initialize(self.layers[i-1])
    
    def fit(self, inputs: np.array):
        for epoch in range(self.epochs):
            output = self.predict(inputs)      
    
    def predict(self, input: np.array):
        input = input
        for layer in self.layers:
            output = layer.feedforward(input)
            input = output
        return output
        
        
        
    
