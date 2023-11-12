import numpy as np
import ActivationFunction

class ReLU(ActivationFunction):
    def __init__(self) -> None:
        super().__init__()
        self.name = "ReLU"
        self.derivative = lambda x: np.where(x > 0, 1, 0)
        self.activation = lambda x: np.maximum(x, 0)
        self.gradient = lambda x: np.where(x > 0, 1, 0)
        
    def __str__(self) -> str:
        return self.name
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.activation(x)
    