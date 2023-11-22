__all__ = ['EarlyStopping']


class EarlyStopping:
    def __init__(self, patience_limit: int = 3):

        self.best_loss = float('inf')
        self.best_epoch = 0
        self.patience = 0
        self.patience_limit = patience_limit
        self.stop = False

    def update(self, model, loss, epoch):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.patience = 0
        else:
            self.patience += 1

        if self.patience > self.patience_limit:
            self.stop = True
            model.stop = True
