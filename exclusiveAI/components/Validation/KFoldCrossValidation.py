from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
from joblib import Parallel, delayed
from exclusiveAI.Composer import Composer
from .ValidationUtils import get_best_model
from tqdm import tqdm
from functools import partial

__all__ = ["validate", "double_validate"]

"""
K-fold cross validation.
"""


def split(x, y_true=None, random_state=None, shuffle=True, n_splits=5):
    """
    Data splitting for cross validation
    Args:
        x: input
        y_true: target
        random_state: random seed
        shuffle: shuffle
        n_splits: number of splits for cross validation

    Raises:
        ValueError: if y_true is not specified and shuffle is True.

    Returns: a tuple containing two lists of indices:
            - The training indices obtained by excluding the validation part in the fold.
            - The validation indices for the current fold.
    """
    if y_true is None and shuffle:
        raise ValueError("y must be specified if shuffle is True")

    indices = np.arange(len(x))  # create an array of indices
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(indices)

    # Calculate the base size of each fold.
    fold_sizes = np.full(n_splits, len(x) // n_splits, dtype=int)
    # The remainder of this division is distributed to ensure that all samples are included.
    fold_sizes[:len(x) % n_splits] += 1

    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        first_part = list(indices[np.concatenate((indices[:start], indices[stop:]))])
        last_part = list(indices[start:stop])
        yield first_part, last_part
        current = stop


def validate_single_config(config, x, y_true, epochs, batch_size, disable_line, random_state, shuffle, n_splits,
                           metric, regression, number_of_initializations):
    best = None
    score = 0
    for train_index, val_index in split(x, y_true, random_state, shuffle, n_splits):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y_true[train_index], y_true[val_index]

        internal_score = 0
        model = None

        for i in range(number_of_initializations):
            tmp_model = Composer(config=config).compose(regression)
            tmp_model.train(inputs=x_train, input_label=y_train, val=x_val, val_labels=y_val,
                            disable_line=disable_line, epochs=epochs, batch_size=batch_size)
            if model is None or np.min(model.history[metric]) > np.min(tmp_model.history[metric]):
                model = tmp_model
            internal_score += np.min(tmp_model.history[metric])
        internal_score /= number_of_initializations

        if best is None or best.get_last()[metric] < score:
            best = model
            score = internal_score
    return score, best, config


def validate(configs, x, y_true, metric='val_mse', max_configs=1, random_state=None, shuffle=True, parallel=True,
             n_splits=5, number_of_initializations=1, epochs=100, assessment=False, batch_size=32,
             disable_line=True, workers=None, regression=False, return_models_history=False):
    evaluate_partial = partial(validate_single_config, x=x.copy(), y_true=y_true.copy(), epochs=epochs,
                               batch_size=batch_size, disable_line=disable_line, random_state=random_state,
                               shuffle=shuffle, n_splits=n_splits, metric=metric, regression=regression,
                               number_of_initializations=number_of_initializations)

    evaluator = ProcessPoolExecutor(max_workers=workers) if parallel else None
    results_func = evaluator.map if evaluator else map

    results = list(tqdm(results_func(evaluate_partial, configs), total=len(configs), desc="Models", colour="white"))

    # results = Parallel(n_jobs=workers)(
    #     delayed(validate_single_config)(config, x.copy(), y_true.copy(), epochs, batch_size, disable_line,
    #                                     random_state, shuffle, n_splits, metric, regression, number_of_initializations)
    #     for config in
    #     tqdm(configs, desc="Models", colour="white")
    # )

    best_models, best_configs = get_best_model(results, max_configs)

    if return_models_history:
        if max_configs > 1:
            return [model.history for model in best_models], best_configs if not assessment else [
                model.get_last()[metric] for model in best_models]
        return best_models[0].history, best_configs[0] if not assessment else best_models[0].get_last()[metric]

    if max_configs > 1:
        return best_configs if not assessment else [model.get_last()[metric] for model in best_models]
    return best_configs[0] if not assessment else best_models[0].get_last()[metric]


def double_validate(configs, x, y_true, metric='val_mse', inner_splits=4, random_state=None,
                    shuffle=True, n_splits=5, number_of_initializations=1,
                    epochs=100, regression=False, return_models_history=False,
                    batch_size=32, disable_line=True, workers=-1):
    """
    Perform a double k-fold cross validation
    Args:
        configs: A dictionary containing configuration parameters for the models
        x: the input
        y_true: the target
        metric: the metric to be used for the cross validation
        inner_splits: the number of splits for the cross validation
        shuffle (bool): true if you want to shuffle the data
        n_splits (int): Number of splits
        number_of_initializations (int): Number of initializations for each model
        random_state (int): random seed
        epochs (int): number of epochs to train the model
        regression (bool): true if you want to use a regression model
        batch_size (int): batch size
        disable_line (bool): whether to disable line
        eps (float): epsilon for model dimension selection
        workers (int): number of workers to use for parallel

    Returns: the best configuration found

    """

    outer_scores = []
    for outer_train_index, outer_val_index in split(x, y_true, shuffle=shuffle, n_splits=n_splits,
                                                    random_state=random_state):
        x_train, x_test = x[outer_train_index], x[outer_val_index]
        y_train, y_test = y_true[outer_train_index], y_true[outer_val_index]

        my_config = validate(configs, x=x_train, y_true=y_train, metric=metric, random_state=random_state,
                             shuffle=False, number_of_initializations=number_of_initializations,
                             n_splits=inner_splits, epochs=epochs, batch_size=batch_size, disable_line=disable_line,
                             regression=regression, workers=workers)
        model = Composer(config=my_config).compose(regression)
        model.train(x_train, y_train, epochs=epochs, batch_size=batch_size, disable_line=disable_line)
        # test error for each external split
        if return_models_history:
            outer_scores.append((model.history, model.evaluate(x_test, y_test)[0]))
        else:
            outer_scores.append(model.evaluate(x_test, y_test)[0])

    return np.mean(outer_scores), outer_scores
