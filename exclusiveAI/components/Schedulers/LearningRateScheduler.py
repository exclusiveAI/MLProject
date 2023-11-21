__all__ = ['LearningRateScheduler']

class LearningRateScheduler:
    def __init__(self, initial_lr: float, lr_scheduler_func):
        self.lr_scheduler_func = lr_scheduler_func