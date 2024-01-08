from exclusiveAI.utils import train_split
from exclusiveAI.Composer import Composer
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from .ValidationUtils import get_best_model
from joblib import Parallel, delayed
import numpy as np
from functools import partial

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


def evaluate_model(config, train, train_target, validation, validation_target, assessment, disable_line, metric,
                   batch_size,
                   epochs, regression, number_of_initializations):
    score = 0
    model = None

    for i in range(number_of_initializations):
        tmp_model = Composer(config=config).compose(regression)
        tmp_model.train(train.copy(), train_target.copy(), None if assessment else validation.copy(),
                        None if assessment else validation_target.copy(), disable_line=disable_line,
                        epochs=epochs, batch_size=batch_size)
        if model is None or np.min(model.history[metric]) > np.min(tmp_model.history[metric]):
            model = tmp_model
        score += np.min(tmp_model.history[metric])
    score /= number_of_initializations

    return score, model, config


def parallel_hold_out(configs, training, training_target, metric=None, num_models=1, regression=False,
                      number_of_initializations=1, epochs=100, batch_size=32, return_models_history=False,
                      disable_line=True, workers: any = None, assessment=False):
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
        num_models: number of models to be returned if all models
        workers: number of workers to use for parallel
        return_models_history: whether to return the model history or not

    Returns: best configuration for the validation task or model assessment for the test task.

    """
    metric = 'val_mse' if not assessment else 'mse' if metric is None else metric
    train, train_target, validation, validation_target = split(training, training_target)

    evaluate_partial = partial(evaluate_model, train=train.copy(), train_target=train_target.copy(),
                               validation=None if assessment else validation.copy(),
                               validation_target=None if assessment else validation_target.copy(),
                               assessment=assessment, disable_line=disable_line, metric=metric, batch_size=batch_size,
                               epochs=epochs, regression=regression,
                               number_of_initializations=number_of_initializations)

    evaluator = ProcessPoolExecutor(max_workers=workers) if workers is None or workers != 1 else None
    results_func = evaluator.map if evaluator else map

    results = list(tqdm(results_func(evaluate_partial, configs), total=len(configs), desc="Models", colour="white"))

    # results = Parallel(n_jobs=workers)(
    #     delayed(evaluate_model)(config, train.copy(), train_target.copy(),
    #                             validation=None if assessment else validation.copy(),
    #                             validation_target=None if assessment else validation_target.copy(),
    #                             assessment=assessment,
    #                             disable_line=disable_line, metric=metric, batch_size=batch_size,
    #                             epochs=epochs, regression=regression,
    #                             number_of_initializations=number_of_initializations)
    #     for config in tqdm(configs, desc="Models", colour="white")
    # )
    best_models, best_configs = get_best_model(results, num_models)

    if return_models_history:
        if num_models > 1:
            return [model.history for model in best_models], best_configs if not assessment else [
                model.evaluate(validation, validation_target)[0] for model in best_models]
        return best_models[0].history, best_configs[0] if not assessment else \
            best_models[0].evaluate(validation, validation_target)[0]

    if num_models > 1:
        return [model.evaluate(validation, validation_target)[0] for model in
                best_models] if assessment else best_configs
    return best_models[0].evaluate(validation, validation_target)[0] if assessment else best_configs[0]


def hold_out(configs, training, training_target, metric: str = None, epochs=100, regression=False,
             number_of_initializations=1, batch_size=32, disable_line=True, assessment=False,
             return_models_history=False, num_models=1):
    return parallel_hold_out(configs=configs, training=training, training_target=training_target, metric=metric,
                             epochs=epochs, regression=regression, number_of_initializations=number_of_initializations,
                             batch_size=batch_size, disable_line=disable_line, assessment=assessment,
                             return_models_history=return_models_history, num_models=num_models, workers=None)
