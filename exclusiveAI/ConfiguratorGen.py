from exclusiveAI.Composer import Composer
import itertools

import numpy as np


class ConfiguratorGen:
    """
    This class is used to generate configurations for the Composer class.

    Args:
        random (bool, optional): If True, generates random configurations. Defaults to False.
        num_of_configurations (int, optional): Number of configurations to generate. Only used if random is True. Defaults to 100.
        output_activation (str, optional): Activation function for the output layer. Defaults to 'linear'.
        regularizations (list, optional): List of regularizations to be used. Defaults to None.
        learning_rates (list, optional): List of learning rates to be used. Defaults to None.
        loss_functions (list, optional): List of loss functions to be used. Defaults to None.
        activation_functions (list, optional): List of activation functions to be used. Defaults to None.
        number_of_units (list, optional): List of number of units to be used. Defaults to None.
        number_of_layers (list, optional): List of number of layers to be used. Defaults to None.
        momentums (list, optional): List of momentums to be used. Defaults to None.
        optimizer (list, optional): List of optimizers to be used. Defaults to None.
        initializers (list, optional): List of initializers to be used. Defaults to None.
        input_shapes (tuple): List of input shapes to be used. Defaults to None.
        callbacks (list, optional): List of callbacks to be used. Defaults to None.
        verbose (bool, optional): Whether to print out the progress of the model. Defaults to False.
        outputs (int, optional): Number of outputs to be used. Defaults to 1.
    """
    def __init__(self, random=False, num_of_configurations=100,
                 output_activation: str = None,
                 regularizations=None,
                 learning_rates=None,
                 loss_functions=None,
                 activation_functions=None,
                 number_of_units=None,
                 number_of_layers=None,
                 momentums=None,
                 optimizer=None,
                 initializers=None,
                 input_shapes=None,
                 callbacks=None,
                 verbose=False,
                 outputs=1):
        if learning_rates is None:
            learning_rates = []
        if regularizations is None:
            regularizations = []
        if loss_functions is None:
            loss_functions = []
        if activation_functions is None:
            activation_functions = []
        if number_of_units is None:
            number_of_units = []
        if number_of_layers is None:
            number_of_layers = []
        if callbacks is None:
            callbacks = []
        if initializers is None:
            initializers = []
        if optimizer is None:
            optimizer = []
        if momentums is None:
            momentums = []
        if output_activation is None:
            output_activation = 'linear'
        self.output_activation = output_activation
        self.input_shapes = input_shapes
        self.callbacks = callbacks
        self.verbose = verbose
        self.outputs = outputs
        self.type = 'random' if random else 'grid'
        self.num_of_configurations = num_of_configurations
        configurations = itertools.product(regularizations,
                                           learning_rates,
                                           loss_functions,
                                           momentums,
                                           optimizer,
                                           number_of_layers,
                                           initializers,
                                           )
        selected_configs = None
        if self.type == 'random':
            configs = list(configurations)
            indices = np.random.permutation(len(configs))
            indices = indices[:self.num_of_configurations]
            selected_configs = [configs[i] for i in indices]
        elif self.type == 'grid':
            selected_configs = list(configurations)

        final_configs = []
        for config in selected_configs:
            internal_config = list(config)
            num_layers = internal_config[5]
            units = list(itertools.product(number_of_units, repeat=num_layers))
            activations = list(itertools.product(activation_functions, repeat=num_layers))
            activations = [list(a) + [self.output_activation] for a in activations]
            for activation in activations:
                for unit in units:
                    local_config = internal_config[:6] + [unit, activation] + internal_config[6:]
                    final_configs.append(local_config)
        self.configs = list(final_configs)
        self.current = -1
        self.max = num_of_configurations if self.type == 'random' else len(self.configs) - 1

    def next(self):
        """
        Returns: the next model/config in the list

        """
        self.current += 1
        if self.verbose:
            print(f"Current configuration: {self.current}")
            print(self.max)
        config = self.configs[self.current]
        composed_model = Composer(regularization=config[0], learning_rate=config[1], loss_function=config[2],
                                  activation_functions=list(config[7]),
                                  num_of_units=list(config[6]), num_layers=config[5], momentum=config[3],
                                  optimizer=config[4],
                                  initializers=config[8],
                                  input_shape=self.input_shapes, callbacks=self.callbacks, verbose=self.verbose,
                                  outputs=self.outputs)
        model = composed_model.compose()
        config = {"regularization": config[0], "learning_rate": config[1], "loss_function": config[2],
                                  "activation_functions": list(config[7]),
                                  "num_of_units": list(config[6]), "num_layers": config[5], "momentum": config[3],
                                  "optimizer": config[4],
                                  "initializers": config[8],
                                  "input_shape": self.input_shapes, "callbacks": self.callbacks, "verbose": self.verbose,
                                  "outputs": self.outputs}
        return model, config

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.max:
            raise StopIteration
        return self.next()
    
    def len(self):
        return self.max
