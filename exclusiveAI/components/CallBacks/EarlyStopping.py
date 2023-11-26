__all__ = ['EarlyStopping']
from ..neural_network import neural_network


class EarlyStopping:
    def __init__(self, patience_limit: int = 3):

        self.best_loss = float('inf')
        self.best_epoch = 0
        self.patience = 0
        self.patience_limit = patience_limit
        self.stop = False

    def __call__(self, model: neural_network):
        val = list(model.history.keys())[1]
        loss = model.history[val][-1]
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
