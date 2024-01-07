from exclusiveAI.utils import train_split
from exclusiveAI.Composer import Composer
from tqdm import tqdm
from .ValidationUtils import get_best_model
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import numpy as np

__all__ = ['parallel_hold_out', "hold_out"]

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


def split(training, training_target, split_size=0.2, shuffle=True, seed=42):
    """
    Split the data into TR and VL/TS
    Returns: TR and VL/TS splits with their target values sets

    """
    train, train_target, validation, validation_target, _, _ = train_split(inputs=training,
                                                                           input_label=training_target,
                                                                           split_size=split_size,
                                                                           shuffle=shuffle,
                                                                           random_state=seed)
    return train, train_target, validation, validation_target


def evaluate_model(config, train, train_target, validation, validation_target, assessment, disable_line, batch_size,
                   epochs, metric, regression, number_of_initializations):
    models = []
    mean_model_score = 0
    best_initialization_score = None
    best_model_initialization = None

    for i in range(number_of_initializations):
        models.append(Composer(config=config).compose(regression))
        models[i].train(train.copy(), train_target.copy(), None if assessment else validation.copy(),
                        None if assessment else validation_target.copy(), disable_line=disable_line,
                        epochs=epochs, batch_size=batch_size)
        if best_initialization_score is None or best_initialization_score > models[i].get_last()[metric]:
            best_initialization_score = models[i].get_last()[metric]
            best_model_initialization = models[i]
        mean_model_score += models[i].get_last()[metric]
    mean_model_score /= number_of_initializations

    model = best_model_initialization

    score = mean_model_score
    std = np.std(model.history[metric])
    return score, std, model, config


def parallel_hold_out(configs, training, training_target, metric=None, eps=1e-2, num_models=1, regression=False,
                      number_of_initializations=1, epochs=100, batch_size=32, return_models_history=False,
                      disable_line=True, workers=None, assessment=False, prefer=''):
    metric = 'val_mse' if not assessment else 'mse' if metric is None else metric
    train, train_target, validation, validation_target = split(training, training_target)

    evaluate_partial = partial(evaluate_model, train=train, train_target=train_target,
                               validation=validation, validation_target=validation_target,
                               assessment=assessment, disable_line=disable_line, metric=metric,
                               batch_size=batch_size, epochs=epochs, regression=regression,
                               number_of_initializations=number_of_initializations)

    if prefer.lower() == "threads":
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(tqdm(executor.map(evaluate_partial, configs),
                                total=len(configs), desc="Models", colour="white"))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(tqdm(executor.map(evaluate_partial, configs),
                                total=len(configs), desc="Models", colour="white"))

    best_models, best_configs = get_best_model(results, eps, num_models)

    if return_models_history:
        if num_models > 1:
            return [model.history for model in best_models], best_configs if not assessment else [
                model.get_last()[metric] for model in best_models]
        return best_models[0].history, best_configs[0] if not assessment else best_models[0].get_last()[metric]

    if num_models > 1:
        return [model.evaluate(validation, validation_target)[0] for model in
                best_models] if assessment else best_configs
    return best_models[0].evaluate(validation, validation_target)[0] if assessment else best_configs[0]


def hold_out(configs, training, training_target, metric: str = None, epochs=100, regression=False,
             number_of_initializations=1, batch_size=32, disable_line=True, assessment=False,
             return_models_history=False,
             eps=1e-2, num_models=1):
    """
    The hold out algorithm
    Args:
        configs (list): List of configurations for models to be evaluated
        training: training data
        training_target: training target
        metric: metric to use (e.g, mse)
        epochs: number of epochs
        regression (bool): whether to use regression or classification
        number_of_initializations (int): number of initializations for each model
        batch_size: batch size
        disable_line: whether to disable the line of model training output
        assessment: whether to assess the model or make model selection
        eps: epsilon for model dimensionality comparison
        num_models: number of models to be returned if all models

    Returns: best configuration for the validation task or model assessment for the test task.

    """
    metric = 'val_mse' if assessment else 'mse' if metric is None else metric
    train, train_target, validation, validation_target = split(training, training_target)

    results = []
    with tqdm(total=len(configs), desc="Models", colour="white") as pbar:
        for config in configs:
            # models = []
            # mean_model_score = 0
            # best_initialization_score = 0
            # best_model_initialization = None
            result = evaluate_model(config, train.copy(), train_target.copy(),
                                    None if assessment else validation.copy(),
                                    None if assessment else validation_target.copy(), assessment, disable_line,
                                    batch_size,
                                    epochs, metric, regression, number_of_initializations)
            # for i in range(number_of_initializations):
            #     models.append(Composer(config=config).compose(regression))
            #     models[i].train(train.copy(), train_target.copy(), None if assessment else validation.copy(),
            #                     None if assessment else validation_target.copy(), disable_line=disable_line,
            #                     epochs=epochs, batch_size=batch_size)
            #     if best_initialization_score is None or best_initialization_score > models[i].get_last()[metric]:
            #         best_initialization_score = models[i].get_last()[metric]
            #         best_model_initialization = models[i]
            #     mean_model_score += models[i].get_last()[metric]
            # mean_model_score /= number_of_initializations
            #
            # model = best_model_initialization
            #
            # score = mean_model_score
            # std = np.std(model.history[metric])
            results.append(result)
            pbar.update(1)
    best_models, best_configs = get_best_model(results, eps, num_models)

    if return_models_history:
        if num_models > 1:
            return [model.history for model in best_models], best_configs if not assessment else [
                model.get_last()[metric] for model in best_models]
        return best_models[0].history, best_configs[0] if not assessment else best_models[0].get_last()[metric]

    if num_models > 1:
        return [model.evaluate(validation, validation_target)[0] for model in
                best_models] if assessment else best_configs
    return best_models[0].evaluate(validation, validation_target)[0] if assessment else best_configs[0]
