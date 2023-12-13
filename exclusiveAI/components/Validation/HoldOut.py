from exclusiveAI.utils import train_split
from exclusiveAI.ConfiguratorGen import ConfiguratorGen
from tqdm import tqdm


class HoldOut:
    """
    Hold-out
    Args:
        models (ConfiguratorGen): a set of models
        input (np.ndarray): input data
        target (np.ndarray): target data
        split_size (float): split size
        shuffle (bool):  true to shuffle data
        seed (int): seed for the random shuffling
        assessment (bool): true to perform model assessment
        debug (bool): true to print debug information
    Attributes:
        best_model (neural_network): the best model found
        best_config (dict): the best configuration found
        models (ConfiguratorGen): a set of models
        input (np.ndarray): input data
        target (np.ndarray): target data
        split_size (float): split size
        shuffle (bool):  true to shuffle data
        seed (int): seed for the random shuffling
        assessment (bool): true to perform model assessment
        debug (bool): true to print debug information
    """

    def __init__(self, models: ConfiguratorGen, input, target, split_size=0.2, shuffle=True, seed=42,
                 assessment: bool = False, debug=False):
        self.best_models = []
        self.best_configs = []
        self.models = models
        self.input = input
        self.target = target
        self.split_size = split_size
        self.shuffle = shuffle
        self.seed = seed
        self.assessment = assessment
        self.debug = debug

    def split(self):
        """
        Split the data into TR and VL/TS
        Returns: TR and VL/TS splits with their target values sets

        """
        train, train_target, validation, validation_target, _, _ = train_split(inputs=self.input,
                                                                               input_label=self.target,
                                                                               split_size=self.split_size,
                                                                               shuffle=self.shuffle,
                                                                               random_state=self.seed)
        return train, train_target, validation, validation_target

    def hold_out(self, metric: str = None, all_models: bool = False, epochs=100, batch_size=32, disable_line=True):
        """
        The hold out algorithm
        Args:
            metric: metric to use (e.g, mse)
            all_models: whether to return the 10 best models or just the best one

        Returns: best configuration for the validation task or model assessment for the test task.

        """
        metric = 'val_mse' if self.assessment else 'mse' if metric is None else metric
        train, train_target, validation, validation_target = self.split()
        with tqdm(total=self.models.len(), desc="Models", colour="white") as pbar:
            for model, config in self.models:
                model.train(train, train_target, None if self.assessment else validation,
                            None if self.assessment else validation_target, disable_line=disable_line, epochs=epochs,
                            batch_size=batch_size)
                if not self.best_models:
                    self.best_models.append(model)
                    self.best_configs.append(config)
                else:
                    if all_models:
                        for i, ith_model in enumerate(self.best_models):
                            if model.get_last()[metric] < ith_model.get_last()[metric]:
                                self.best_models.insert(i, model)
                                self.best_configs.insert(i, config)
                        if len(self.best_models) > 100:
                            self.best_models = self.best_models[:100]
                            self.best_configs = self.best_configs[:100]
                    else:
                        if model.get_last()[metric] < self.best_models[0].get_last()[metric]:
                            self.best_models[0] = model
                            self.best_configs[0] = config
                pbar.update(1)
        if all_models:
            return [model.evaluate(validation, validation_target) for model in self.best_models] if self.assessment else self.best_configs
        return self.best_models[0].evaluate(validation, validation_target) if self.assessment else self.best_configs[0]