__all__ = ['EarlyStoppingCallback']
from ..NeuralNetwork import NeuralNetwork


class EarlyStoppingCallback:
    """
    Implements early stopping
    """
    def __init__(self, patience_limit: int = 3, metric: str='val_mse'):

        self.best_loss = float('inf')
        self.best_epoch = 0
        self.patience = 0
        self.patience_limit = patience_limit
        self.metric = metric
        self.stop = False

    def __call__(self, model: NeuralNetwork):
        if self.metric=='val_mse':
            # check if val_mse in history
            if 'val_mse' not in model.history:
                self.metric = 'mse'
        loss = model.history[self.metric][-1]
        if model.curr_epoch == 0:
            self.best_loss = loss
            self.best_epoch = 0
            self.patience = 0
            return
        epoch = model.curr_epoch
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.patience = 0
        else:
            self.patience += 1

        if self.patience > self.patience_limit:
            self.stop = True
            model.early_stop = True

    def reset(self):
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.patience = 0
        self.stop = False
        self.metric = 'val_mse'
