__all__ = ['EarlyStoppingCallback']

from ..NeuralNetwork import NeuralNetwork


class EarlyStoppingCallback:
    """
    Implements early stopping
    """

    def __init__(self, eps: float = 1e-4, patience_limit: int = 50, restore_weights: bool = False, penalty=False,
                 metric: str = 'val_mse', **kwargs):
        self.restore_weights = restore_weights
        self.patience_limit = patience_limit
        self.penalty = penalty
        self.metric = metric
        self.stop = False
        self.eps = eps

    def __call__(self, model: NeuralNetwork):
        if self.metric == 'val_mse':
            # check if val_mse in history
            if 'val_mse' not in model.history:
                self.metric = 'mse'
        loss = model.history[self.metric][-1]
        if model.curr_epoch == 0:
            model.best_loss = loss
            model.best_epoch = 0
            model.best_weights = model.get_weights()
            model.patience = 0
            return
        epoch = model.curr_epoch
        if model.best_loss - loss > self.eps:
            model.best_loss = loss
            model.best_epoch = epoch
            model.patience = 0
            model.best_weights = model.get_weights()
        else:
            model.patience += 1
            if self.penalty and model.best_loss - loss < 0:
                model.patience += 1

        if model.patience > self.patience_limit:
            self.stop = True
            model.early_stop = True
            if self.restore_weights:
                model.set_weights(model.best_weights)
                model.history = {key: value[:model.best_epoch] for key, value in model.history.items()}

    def reset(self):
        self.stop = False
        self.metric = 'val_mse'

    def close(self):
        self.reset()
