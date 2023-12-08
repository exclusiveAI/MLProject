__all__ = ['Optimizer']

from exclusiveAI.components.Schedulers import LearningRateScheduler


class Optimizer:
    """
    Optimizer class
    Args:
        learning_rate (float): learning rate
        lr_scheduler (LearningRateScheduler): learning rate scheduler
    Attributes:
        learning_rate (float): learning rate
        lr_scheduler (LearningRateScheduler): learning rate scheduler
        old_dw (list): list of old deltas of weights
    """
    def __init__(self, lr_scheduler=None, **kwargs):
        self.lr_scheduler = lr_scheduler
        self.old_dw = []

    def calulate_deltas(self, model, y_true, x):
        """
        Calculate deltas
        Args:
            model: the current model
            y_true: the target value
            x: the input value

        Returns:
            deltas: the deltas of weights
        """
        model.predict(x)

        layers = list(reversed(model.layers)) #to start from output layer
        output_layer = layers.pop(0)
        deltas = []
        deltas.insert(0, output_layer.backpropagate(y_true))

        for layer in layers:
            deltas.insert(0, layer.backpropagate())
        return deltas

    def update_lr(self):
        self.learning_rate = self.lr_scheduler.update(self.learning_rate)

    def update(self, model, y_true, x):
        pass
