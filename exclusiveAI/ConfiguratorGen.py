from itertools import product
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial


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
                 show_line=False,
                 max_configs=100,
                 verbose=False,
                 random=False,
                 beta2=None,
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
        self.beta2 = beta2

        self.type = 'random' if random else 'grid'
        self.num_of_configurations = max_configs

        if beta2:
            configurations = product(regularizations,
                                     learning_rates,
                                     loss_function,
                                     momentums,
                                     optimizer,
                                     number_of_layers,
                                     initializers,
                                     self.nesterov,
                                     beta2
                                     )
        else:
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

        final_configs = self.par_combinations(selected_configs, self.activation_functions, self.number_of_units,
                                              self.output_activation, self.input_shapes, self.callbacks, self.verbose,
                                              self.outputs)
        # with tqdm(total=len(selected_configs) * number_of_initializations, desc="2nd for", colour="white",
        #           disable=not show_line) as pbar:
        #     final_configs = []
        #     for config in selected_configs:
        #         internal_config = list(config)
        #         num_layers = internal_config[5]
        #         product_unit_activation = self.units_activations_per_layer_combinations(self.activation_functions,
        #                                                                                 self.number_of_units,
        #                                                                                 num_layers)
        #         for unit, activation in product_unit_activation:
        #             local_config = internal_config[:6] + [unit, activation] + internal_config[6:]
        #             final_configs.append(local_config)
        #         pbar.update(1)

        if number_of_initializations > 1:
            with tqdm(total=len(final_configs) * number_of_initializations, desc="1st for", colour="white",
                      disable=not show_line) as pbar:
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

    @staticmethod
    def add_units_activations(config, activation_functions, number_of_units, pbar):
        final_configs = []
        internal_config = list(config)
        num_layers = internal_config[5]
        product_unit_activation = ConfiguratorGen.units_activations_per_layer_combinations(activation_functions,
                                                                                           number_of_units,
                                                                                           num_layers)
        for unit, activation in product_unit_activation:
            local_config = internal_config[:6] + [unit, activation] + internal_config[6:]
            final_configs.append(local_config)
        pbar.update(1)
        return final_configs

    @staticmethod
    def par_combinations(selected_configs, activation_functions, number_of_units,
                         output_activation, input_shapes, callbacks, verbose, outputs):

        batch_size = 1000  # You can adjust the batch size based on your requirements
        with tqdm(total=len(selected_configs), desc="Appending") as pbar:
            with ThreadPoolExecutor(max_workers=8) as executor:
                batches = [selected_configs[i:i + batch_size] for i in range(0, len(selected_configs), batch_size)]

                batch_results = []
                for batch in batches:
                    evaluate_partial_batch = partial(ConfiguratorGen.add_units_activations,
                                                     activation_functions=activation_functions,
                                                     number_of_units=number_of_units, pbar=pbar)
                    batch_results.append(executor.map(evaluate_partial_batch, batch))

        output = []
        counter = 0
        final_configs = []
        for batch in batch_results:
            for element in batch:
                for config in element:
                    final_configs.append(config)
        with tqdm(total=len(final_configs), desc="Appending") as pbar:
            for config in final_configs:
                config = {"regularization": config[0], "learning_rate": config[1], "loss_function": config[2],
                          "activation_functions": list(config[7]), "output_activation": output_activation,
                          "num_of_units": list(config[6]), "num_layers": config[5], "momentum": config[3],
                          "optimizer": config[4],
                          "initializers": config[8], "nesterov": True if config[9] == 'True' else False,
                          "beta2": config[10] if len(config)>10 else 0,
                          "input_shape": input_shapes, "callbacks": callbacks, "verbose": verbose,
                          "outputs": outputs, "model_name": 'Model' + str(counter)}
                output.append(config)
                counter += 1
                pbar.update(1)
        return output

    def next(self):
        """
        Returns: the next model/config in the list

        """
        if self.verbose:
            print(f"Current configuration: {self.current} of {self.max}")
        config = self.configs[self.current]
        # config = {"regularization": config[0], "learning_rate": config[1], "loss_function": config[2],
        #           "activation_functions": list(config[7]), "output_activation": self.output_activation,
        #           "num_of_units": list(config[6]), "num_layers": config[5], "momentum": config[3],
        #           "optimizer": config[4],
        #           "initializers": config[8], "nesterov": True if config[9] == 'True' else False,
        #           "input_shape": self.input_shapes, "callbacks": self.callbacks, "verbose": self.verbose,
        #           "outputs": self.outputs, "model_name": 'Model' + str(self.current)}

        return config

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

    @staticmethod
    def units_activations_per_layer_combinations(activation_functions, units, layers):
        return product(product(units, repeat=layers), product(activation_functions, repeat=layers))

    def get_configs(self):
        return self.configs
        # tmp_configs = []
        # for i, config in enumerate(self.configs):
        #     t_config = {"regularization": config[0], "learning_rate": config[1], "loss_function": config[2],
        #                 "activation_functions": list(config[7]), "output_activation": self.output_activation,
        #                 "num_of_units": list(config[6]), "num_layers": config[5], "momentum": config[3],
        #                 "optimizer": config[4],
        #                 "initializers": config[8], "nesterov": True if config[9] == 'True' else False,
        #                 "input_shape": self.input_shapes, "callbacks": self.callbacks, "verbose": self.verbose,
        #                 "outputs": self.outputs, "model_name": 'Model' + str(i)}
        #     tmp_configs.append(t_config)
        # return tmp_configs
