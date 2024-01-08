from exclusiveAI.components.ActivationFunctions import *
from exclusiveAI.components.LossFunctions import *
from exclusiveAI.components.Initializers import *
from exclusiveAI.components.CallBacks import *
from exclusiveAI.components.Optimizers import *
from exclusiveAI.components.Layers import *
from exclusiveAI.components import NeuralNetwork

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
    'crosscorrelation': CrossCorrelation,
}

ActivationFunctionsNames = {
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'relu': ReLU,
    'linear': Linear,
    'softmax': Softmax,
}

OptimizersNames = {
    'sgd': SGD,
    'adam': Adam,
}


class Composer:
    """
    Compose the building block to build the model.
    Args:
        regularization (float): The regularization parameter to be used for the model.
        learning_rate (float): The learning rate to be used for the model.
        loss_function (str): The loss function to be used for the model.
        activation_functions (list): The activation functions to be used for the model.
        num_of_units (list): The number of units in each layer.
        num_layers (int): The number of layers in the model.
        momentum (float): The momentum parameter to be used for the model.
        optimizer (str): The optimizer to be used for the model.
        initializers (list): The initializers to be used for the model.
        input_shape (tuple): The input shape of the model.
        callbacks (list): The callbacks to be used for the model.
        verbose (bool): Whether to print out the progress of the model.
        outputs (int): The number of outputs of the model.
        config (dict): The configuration of the model. Note: you can take a config dict in input to build the model instead of each single param.
    Attributes:

    """

    def __init__(self,
                 regularization: float = None,
                 learning_rate: float = None,
                 loss_function: str = None,
                 activation_functions=None,
                 output_activation=None,
                 num_of_units: list = None,
                 model_name: str = 'test',
                 num_layers: int = None,
                 nesterov: bool = False,
                 momentum: float = None,
                 optimizer: str = None,
                 beta1: float = None,
                 beta2: float = None,
                 initializers=None,
                 eps: float = None,
                 input_shape=None,
                 callbacks=None,
                 verbose=False,
                 outputs=1,
                 config: {} = None,
                 ):
        if config is not None:
            if isinstance(config, dict):
                regularization = config.get('regularization', regularization)
                learning_rate = config.get('learning_rate', learning_rate)
                loss_function = config.get('loss_function', loss_function)
                activation_functions = config.get('activation_functions', activation_functions)
                output_activation = config.get('output_activation', output_activation)
                num_of_units = config.get('num_of_units', num_of_units)
                num_layers = config.get('num_layers', num_layers)
                momentum = config.get('momentum', momentum)
                optimizer = config.get('optimizer', optimizer)
                initializers = config.get('initializers', initializers)
                callbacks = config.get('callbacks', callbacks)
                nesterov = config.get('nesterov', nesterov)
                verbose = config.get('verbose', verbose)
                outputs = config.get('outputs', outputs)
                input_shape = config.get('input_shape', input_shape)
                model_name = config.get('model_name', model_name)
            # else:
            #     regularization = config[0]
            #     learning_rate = config[1]
            #     loss_function = config[2]
            #     activation_functions = list(config[7])
            #     output_activation = config
            #     t_config = {"regularization": config[0], "learning_rate": config[1], "loss_function": config[2],
            #                 "activation_functions": list(config[7]), "output_activation": self.output_activation,
            #                 "num_of_units": list(config[6]), "num_layers": config[5], "momentum": config[3],
            #                 "optimizer": config[4],
            #                 "initializers": config[8], "nesterov": True if config[9] == 'True' else False,
            #                 "input_shape": self.input_shapes, "callbacks": self.callbacks, "verbose": self.verbose,
            #                 "outputs": self.outputs, "model_name": 'Model' + str(i)}
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
            optimizer = SGD()
        if initializers is None:
            initializers = [Gaussian()]
        if callbacks is None:
            callbacks = []
        if loss_function is None:
            loss_function = MeanSquaredError()
        if activation_functions is None:
            activation_functions = [Sigmoid()]
        if output_activation is None:
            output_activation = Sigmoid()
        if not isinstance(activation_functions, list):
            activation_functions = [activation_functions]
        if not isinstance(initializers, list):
            initializers = [initializers]
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        if len(initializers) > 1:
            if len(initializers) != num_layers:
                raise ValueError("Parameter initializers must have the same length as num_layers")
            else:
                self.manyInitializers = True
        else:
            self.manyInitializers = False
        if len(activation_functions) - 1 > 1:
            if len(activation_functions) != num_layers:
                print(activation_functions, num_layers, config)
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

        self.callbacks = [self.str_to_callback(callback, model_name)
                          for callback in callbacks]

        self.loss_function = LossFunctionsNames[loss_function.lower()]() \
            if isinstance(loss_function, str) else loss_function

        self.activation_functions = [ActivationFunctionsNames[activation_function.lower()]()
                                     if isinstance(activation_function, str) else activation_function
                                     for activation_function in activation_functions]

        self.output_activation = ActivationFunctionsNames[output_activation.lower()]() if isinstance(output_activation,
                                                                                                     str) else output_activation

        self.optimizer = OptimizersNames[optimizer.lower()](regularization=regularization,
                                                            learning_rate=learning_rate,
                                                            momentum=momentum,
                                                            nesterov=nesterov,
                                                            beta1=beta1,
                                                            beta2=beta2,
                                                            eps=eps
                                                            ) \
            if isinstance(optimizer, str) else optimizer

        self.num_of_units = num_of_units
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.output_units = outputs
        self.verbose = verbose

    def compose(self, regression: bool = False) -> NeuralNetwork.NeuralNetwork:
        """
        Compose the building block to build the model.
        Returns:
            model: The model composed.
        """
        layers = []
        input_layer = InputLayer(self.input_shape[0], self.input_shape[-1])
        layers.append(input_layer)
        for i in range(self.num_layers):
            layers.append(Layer(self.num_of_units[i], self.initializers[i if self.manyInitializers else 0],
                                self.activation_functions[i if self.manyActivations else 0]))
        output_layer = OutputLayer(units=self.output_units, activation_function=self.output_activation,
                                   initializer=self.initializers[-1], loss_function=self.loss_function)
        layers.append(output_layer)

        model = NeuralNetwork.NeuralNetwork(optimizer=self.optimizer,
                                            callbacks=self.callbacks,
                                            metrics=['mse', 'mae', 'mee', 'binary_accuracy'] if not regression else [
                                                'mse', 'mae', 'mee'],
                                            layers=layers,
                                            verbose=self.verbose)

        return model

    @staticmethod
    def str_to_callback(callback, model_name):
        if isinstance(callback, str):
            if '_' in callback:
                call = callback.split('_')
                local_eps = 1e-4
                patience_limit = 50
                restore_weights = False
                penalty = False
                for i, arg in enumerate(call[1:]):
                    if i == 0:
                        local_eps = float(arg)
                    elif i == 1:
                        patience_limit = int(arg)
                    elif i == 2:
                        restore_weights = True if arg.lower() == 'true' else False
                    elif i == 3:
                        penalty = True if arg.lower() == 'true' else False
                return CallbacksNames[call[0].lower()](run_name=model_name, eps=local_eps, penalty=penalty,
                                                       patience_limit=patience_limit, restore_weights=restore_weights)
            else:
                return CallbacksNames[callback.lower()](run_name=model_name)
        else:
            return callback
