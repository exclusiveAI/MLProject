__all__ = ['wandb_Logger']

import wandb


class wandb_Logger:
    def __init__(self):
        self.name = 'wandb'
        self.config = None
        self.project = None
        self.run = None
        self.log_dict = {}
        self.log_list = []

    @staticmethod
    def init(project, name, config):
        wandb.init(project=project, name=name, config=config)
        # define our custom x axis metric
        wandb.define_metric("train/step")
        # set all other train/ metrics to use this step
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("val/*", step_metric="train/step")

    def __call__(self, model, *args, **kwargs):
        self.log_dict = {
            'train/loss':  model.metrics['train_loss'],
            'train/acc': model.metrics['train_acc'],
            'val/loss': model.metrics['val_loss'],
            'train/step': model.metrics['epoch'],
            'val/acc': model.metrics['val_acc'],
        }
        wandb.log(self.log_dict)