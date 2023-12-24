__all__ = ['EarlyStoppingCallback']

from ..NeuralNetwork import NeuralNetwork


class EarlyStoppingCallback:
    """
    Implements early stopping
    """

    def __init__(self, eps: float = 1e-4, patience_limit: int = 10, metric: str = 'val_mse',
                 restore_weights: bool = False, **kwargs):
        self.restore_weights = restore_weights
        self.patience_limit = patience_limit
        self.best_loss = float('inf')
        self.best_weights = None
        self.metric = metric
        self.best_epoch = 0
        self.patience = 0
        self.stop = False
        self.eps = eps

    def __call__(self, model: NeuralNetwork):
        if self.metric == 'val_mse':
            # check if val_mse in history
            if 'val_mse' not in model.history:
                self.metric = 'mse'
        loss = model.history[self.metric][-1]
        if model.curr_epoch == 0:
            self.best_loss = loss
            self.best_epoch = 0
            self.best_weights = model.get_weights()
            self.patience = 0
            return
        epoch = model.curr_epoch
        if self.best_loss - loss > self.eps:
            self.best_loss = loss
            self.best_epoch = epoch
            self.patience = 0
            self.best_weights = model.get_weights()
        else:
            self.patience += 1

        if self.patience > self.patience_limit:
            self.stop = True
            model.early_stop = True
            if self.restore_weights:
                model.set_weights(self.best_weights)
                model.history = model.history[:self.best_epoch]
                model.curr_epoch = self.best_epoch

    def reset(self):
        self.best_loss = float('inf')
        self.best_weights = None
        self.best_epoch = 0
        self.patience = 0
        self.stop = False
        self.metric = 'val_mse'

    def close(self):
        self.reset()
