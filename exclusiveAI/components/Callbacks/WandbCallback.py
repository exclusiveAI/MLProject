__all__ = ['WandbCallback']

import wandb


class WandbCallback:
    def __init__(self, run_name, project='exclusiveAI', config=None):
        self.name = 'wandb'
        self.run = ''
        self.config = config
        self.project = None
        self.run = None
        self.log_dict = {}
        self.log_list = []
        wandb.init(project=project, name=run_name, config=config)
        wandb.define_metric("train/step")
        # set all other train/ metrics to use this step
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("val/*", step_metric="train/step")

    def __call__(self, model, *args, **kwargs):
        for name in model.history:
            if name.split('_')[0] == 'val':
                self.log_dict['val/' + name.split('_')[-1]] = model.history[name][-1]
            else:
                self.log_dict['train/' + name] = model.history[name][-1]
            self.log_dict['train/step'] = model.curr_epoch
        wandb.log(self.log_dict)
