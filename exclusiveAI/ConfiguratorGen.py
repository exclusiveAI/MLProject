from exclusiveAI.Composer import Composer
from itertools import product
from tqdm import tqdm
import numpy as np
import pickle


class ConfiguratorGen:
    """
    This class is used to generate configurations for the Composer class.

    Args:
        random (bool, optional): If True, generates random configurations. Defaults to False.
        max_configs (int, optional): Number of configurations to generate. Only used if random is True. Defaults to 100.
        output_activation (str, optional): Activation function for the output layer. Defaults to 'linear'.
        regularizations (list, optional): List of regularizations to be used. Defaults to None.
        learning_rates (list, optional): List of learning rates to be used. Defaults to None.
        loss_function (str, optional): Loss function to be used. Default to linear.
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

    def __init__(self,
                 output_activation: str | list,
                 loss_function: str | list,
                 optimizer: str | list,
                 activation_functions: list,
                 input_shapes: int | tuple,
                 number_of_layers: list,
                 number_of_units: list,
                 learning_rates: list,
                 initializers: list,
                 callbacks: list,
                 number_of_initializations=1,
                 regularizations=None,
                 nesterov=False,
                 momentums=None,
                 max_configs=100,
                 verbose=False,
                 random=False,
                 outputs=1
                 ):
        """

        """
        if regularizations is None:
            regularizations = [0]
        if optimizer is None or optimizer == []:
            optimizer = ['sgd']
        if momentums is None:
            momentums = [0]
        self.output_activation = output_activation \
            if isinstance(output_activation, list) else output_activation
        self.loss_function = loss_function \
            if isinstance(loss_function, list) else [loss_function]
        self.optimizer = optimizer \
            if isinstance(optimizer, list) else [optimizer]

        self.activation_functions = activation_functions
        self.number_of_layers = number_of_layers
        self.number_of_units = number_of_units
        self.regularizations = regularizations
        self.learning_rates = learning_rates
        self.initializers = initializers
        self.input_shapes = input_shapes
        self.callbacks = callbacks
        self.nesterov = ["True", "False"] if nesterov else ["False"]
        self.verbose = verbose
        self.outputs = outputs

        self.type = 'random' if random else 'grid'
        self.num_of_configurations = max_configs

        configurations = product(regularizations,
                                 learning_rates,
                                 loss_function,
                                 momentums,
                                 optimizer,
                                 number_of_layers,
                                 initializers,
                                 self.nesterov,
                                 )

        selected_configs = list(configurations)

        with tqdm(total=len(selected_configs) * number_of_initializations, desc="2nd for", colour="white") as pbar:
            final_configs = []
            for config in selected_configs:
                internal_config = list(config)
                num_layers = internal_config[5]
                units = self.units_per_layer_combinations(self.number_of_units, num_layers)
                activations = self.activation_per_layer(self.activation_functions, num_layers)
                for activation in activations:
                    for unit in units:
                        local_config = internal_config[:6] + [unit, activation] + internal_config[6:]
                        final_configs.append(local_config)
                pbar.update(1)

        if number_of_initializations > 1:
            with tqdm(total=len(final_configs) * number_of_initializations, desc="1st for", colour="white") as pbar:
                tmp_configurations = []
                for config in final_configs:
                    for i in range(number_of_initializations):
                        tmp_configurations.append(config)
                        pbar.update(1)

                final_configs = tmp_configurations

        if self.type == 'random':
            indices = np.random.permutation(len(final_configs))
            indices = indices[:self.num_of_configurations] if indices.size > self.num_of_configurations else indices
            final_configs = [final_configs[i] for i in indices]
        self.configs = list(final_configs)
        self.current = -1
        self.max = max(max_configs, len(self.configs)) if self.type == 'random' else len(self.configs)

    def next(self):
        """
        Returns: the next model/config in the list

        """
        if self.verbose:
            print(f"Current configuration: {self.current} of {self.max}")
        config = self.configs[self.current]
        config = {"regularization": config[0], "learning_rate": config[1], "loss_function": config[2],
                  "activation_functions": list(config[7]), "output_activation": self.output_activation,
                  "num_of_units": list(config[6]), "num_layers": config[5], "momentum": config[3],
                  "optimizer": config[4],
                  "initializers": config[8], "nesterov": True if config[9] == 'True' else False,
                  "input_shape": self.input_shapes, "callbacks": self.callbacks, "verbose": self.verbose,
                  "outputs": self.outputs, "model_name": 'Model' + str(self.current)}
        composed_model = Composer(config=config)
        model = composed_model.compose()

        return model, config

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current < self.max:
            return self.next()
        else:
            raise StopIteration

    def len(self):
        return self.max

    def save(self, filename="configs.pkl"):
        """
        Save self.configs as a pickle file.

        Args:
            filename (str, optional): The name of the pickle file. Defaults to "configs.pkl".
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def units_per_layer_combinations(units, layers):
        tmp_units = units
        result = [list(x) for x in product(tmp_units, repeat=layers)]
        return result

    @staticmethod
    def activation_per_layer(activation_functions, layers):
        tmp_activations = activation_functions
        result = [list(x) for x in list(product(tmp_activations, repeat=layers))]
        return result
