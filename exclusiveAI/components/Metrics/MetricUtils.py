from . import MAE
from . import MEE
from . import MSE

__all__ = ["stringToMetric", "initializeHistory", "addToHistory", "printHistory"]

MATCH = {
    "mse": MSE,
    "mae": MAE,
    "mee": MEE,
}


def stringToMetric(name):
    if name.lower() in MATCH:
        return MATCH[name.lower()]()
    else:
        # Error
        raise ValueError("Unknown metric name: " + name)


def initializeHistory(model, val: bool):
    model.metrics = [stringToMetric(metric) if isinstance(metric, str) else metric for metric in model.metrics]
    model.history = {}
    for metric in model.metrics:
        model.history[metric.name] = []
        if val:
            model.history["val_" + metric.name] = []
    print(model.history)


def addToHistory(model, y_train_pred, y_train_true, y_val_pred, y_val_true):
    for metric in model.metrics:
        model.history[metric.name].append(metric(y_train_true, y_train_pred))
        if y_val_pred is not None and y_val_true is not None:
            model.history["val_" + metric.name].append(metric(y_val_true, y_val_pred))


def printHistory(model, val: bool):
    for metric in model.history:
        print(metric, model.history[metric])
        if val:
            print("val_" + metric, model.history["val_" + metric])
