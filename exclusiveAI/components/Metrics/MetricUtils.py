from . import MAE
from . import MEE
from . import MSE
from . import BinaryAccuracy

__all__ = ["string_to_metric", "initialize_history", "add_to_history", "print_history", "calculate"]

MATCH = {
    "mse": MSE,
    "mae": MAE,
    "mee": MEE,
    "binary_accuracy": BinaryAccuracy,
}


def string_to_metric(name):
    """
    Args:
        name (str): name of the metric.
    Raises:
        ValueError: the input string doesn't match any available metric.
    Returns:
        object (Metric): returns a metric matching the input string.
    """
    if name.lower() in MATCH:
        return MATCH[name.lower()]()
    else:
        # Error
        raise ValueError("Unknown metric name: " + name)


def initialize_history(model, val: bool):
    """
    Initializes the metric history of a given model.
    Args:
        model: the model for which to initialize the history.
        val: true if you want to add validation metrics too.
    """
    model.metrics = [string_to_metric(metric) if isinstance(metric, str) else metric for metric in model.metrics]
    model.history = {}
    for metric in model.metrics:
        model.history[metric.name] = []
        if val:
            model.history["val_" + metric.name] = []


def add_to_history(model, y_train_pred, y_train_true, y_val_pred, y_val_true):
    """
    Calculates and adds to the history of a given model the metrics calculated on the training and validation (if specified) data.
    Args:
        model: the model for which to add the metric to its history
        y_train_pred: predicted value by the model for TR
        y_train_true: corresponding target value
        y_val_pred: predicted value by the model for VL
        y_val_true: corresponding target value
    """
    for metric in model.metrics:
        if y_train_true.shape == (y_train_true.shape[0],):
            y_train_true = y_train_true.reshape(-1, 1)
        model.history[metric.name].append(metric(y_train_true, y_train_pred))
        if y_val_pred is not None and y_val_true is not None:
            if y_val_true.shape == (y_val_true.shape[0],):
                y_val_true = y_val_true.reshape(-1, 1)
            model.history['val_' + metric.name].append(metric(y_val_true, y_val_pred))


def print_history(model, val: bool):
    """
    Prints the history of a given model
    Args:
        model: the given model for which to print the metrics history
        val: true if you want to print the validation metrics too
    """
    for metric in model.history:
        print(metric, model.history[metric])
        if val:
            print("val_" + metric, model.history["val_" + metric])


def calculate(func, target, predicted):
    """
    Calculates the metrics specified by the input string.
    Args:
        func (str): the metric name
        target: target value
        predicted: predicted value by the model

    Raises:
        ValueError: if the metric name is unknown

    """
    func = func.lower()
    if func not in MATCH:
        raise ValueError("Unknown metric name: " + func)
    func = MATCH[func]()
    if target.shape == (target.shape[0],):
        target = target.reshape(-1, 1)
    return func(target, predicted)
