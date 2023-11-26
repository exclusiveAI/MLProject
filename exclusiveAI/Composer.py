from exclusiveAI.components.ActivationFunctions import *
from exclusiveAI.components.LossFunctions import *
from exclusiveAI.components.Initializers import *
from exclusiveAI.components.Callbacks import *
from exclusiveAI.components.Optimizers import *
from exclusiveAI.components.Layers import *
from exclusiveAI import neural_network

InitializersNames = {
    'gaussian': Gaussian,
    'uniform': Uniform,
}

CallbacksNames = {
    'earlystopping': EarlyStoppingCallback,
    'wandb': WandbCallback,
}

LossFunctionsNames = {
    'meansquarederror': MeanSquaredError,
    'meaneuclideandistance': MeanEuclideanDistance,
    'mse': MeanSquaredError,
    'mee': MeanEuclideanDistance,
}

ActivationFunctionsNames = {
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'relu': ReLU,
    'linear': Linear,
}

OptimizersNames = {
    'sgd': SGD,
    'adam': Adam,
    'nesterovsgd': NesterovSGD,
}


class Composer:
    def __init__(self,
                 regularization: float = None,
                 learning_rate: float = None,
                 loss_function: str = None,
                 activation_functions=None,
                 num_of_units: list = None,
                 num_layers: int = None,
                 momentum: float = None,
                 optimizer: str = None,
                 # beta1: float = None,
                 # beta2: float = None,
                 initializers=None,
                 # eps: float = None,
                 input_shape=None,
                 callbacks=None,
                 verbose=False,
                 outputs=1
                 ):
        self.optimizer = optimizer
        if input_shape is None:
            # Error can't initialize
            raise ValueError("Parameter input_shape can't be None")
        if num_layers is None:
            raise ValueError("Parameter num_layers can't be None")
        if num_of_units is None:
            raise ValueError("Parameter num_of_units can't be None")
        if len(num_of_units) != num_layers:
            raise ValueError("Parameter num_of_units must have the same length as num_layers")
        if optimizer is None:
            if learning_rate is None:
                learning_rate = 0.01
            if regularization is None:
                regularization = 0.01
            if momentum is None:
                momentum = 0.9
            optimizer = SGD(learning_rate=learning_rate, momentum=momentum, regularization=regularization)
        if initializers is None:
            initializers = [Gaussian()]
        if callbacks is None:
            callbacks = []
        if loss_function is None:
            loss_function = MeanSquaredError()
        if activation_functions is None:
            activation_functions = [Sigmoid(), Linear()]
        if not isinstance(activation_functions, list):
            activation_functions = [activation_functions]
        if not isinstance(initializers, list):
            initializers = [initializers]
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        if len(activation_functions) == 1:
            activation_functions.insert(0, Sigmoid())
        if len(initializers) > 1:
            if len(initializers) != num_layers:
                raise ValueError("Parameter initializers must have the same length as num_layers")
            else:
                self.manyInitializers = True
        else:
            self.manyInitializers = False
        if len(activation_functions)-1 > 1:
            if len(activation_functions) != num_layers+1:
                print(activation_functions, num_layers)
                raise ValueError("Parameter activation_functions must have the same length as num_layers")
            else:
                self.manyActivations = True
        else:
            self.manyActivations = False

        # Get each initializer for each layer
        # If one, every layer will have the same initializer
        # If more than one, each layer will have its own initializer and the # of initializers must be equal to # of layer
        self.initializers = [InitializersNames[initializer.lower()]() if isinstance(initializer, str) else initializer
                             for initializer in initializers]

        self.callbacks = [CallbacksNames[callback.lower()]() if isinstance(callback, str) else callback
                          for callback in callbacks]

        self.loss_function = LossFunctionsNames[loss_function.lower()]() \
            if isinstance(loss_function, str) else loss_function

        self.activation_functions = [ActivationFunctionsNames[activation_function.lower()]()
                                     if isinstance(activation_function, str) else activation_function
                                     for activation_function in activation_functions]
        if (learning_rate is None or regularization is None or momentum is None and isinstance(optimizer, str) and
                (optimizer.lower() == 'sgd' or optimizer.lower() == 'nesterovsgd')):
            raise ValueError("Parameters learning_rate, regularization and momentum can't be None if Optimizer: ",
                             optimizer)
        else:
            self.optimizer = OptimizersNames[optimizer.lower()](learning_rate=learning_rate, regularization=regularization, momentum=momentum) \
                if isinstance(optimizer, str) else optimizer
        self.num_of_units = num_of_units
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.output_units = outputs
        self.verbose = verbose

    def compose(self):
        layers = []
        input_layer = InputLayer(self.input_shape)
        layers.append(input_layer)
        for i in range(self.num_layers):
            layers.append(Layer(self.num_of_units[i], self.initializers[i if self.manyInitializers else 0],
                                self.activation_functions[i if self.manyActivations else 0]))
        output_layer = OutputLayer(units=self.output_units, activation_function=self.activation_functions[-1],
                                   initializer=self.initializers[-1], loss_function=self.loss_function)
        layers.append(output_layer)

        model = neural_network.neural_network(optimizer=self.optimizer,
                                              callbacks=self.callbacks,
                                              metrics=['mse', 'mae', 'mee'],
                                              layers=layers,
                                              verbose=self.verbose)

        return model