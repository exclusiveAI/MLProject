import numpy as np
from exclusiveAI.components.LossFunctions import LossFunction
from exclusiveAI.components.Optimizers import Optimizer
from exclusiveAI.components.Metrics import MetricUtils
from exclusiveAI import utils
from tqdm import tqdm


class neural_network:
    def __init__(self,
                 layers: list,
                 optimizer: Optimizer,
                 callbacks: list,
                 metrics: [],
                 verbose: bool = False,
                 ):
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.layers = layers
        self.verbose = verbose
        self.early_stop = False
        self.name = 'Model ' + str(len(self.layers))
        self.curr_epoch = 0
        self.metrics = metrics
        self.history = {}

        self.initialize()

    def initialize(self):
        self.layers[0].initialize(name='Input', verbose=self.verbose)
        for i, layer in enumerate(self.layers[1:]):
            layer.initialize(self.layers[i], name=('Layer' + str(i)), verbose=self.verbose)

    def train(self,
              inputs: np.array,
              input_label: np.array,
              val: np.array = None,
              val_labels: np.array = None,
              epochs=1000, batch_size=32, name:str='', thread: int=0):
        # check if both val and val_label are provided
        if val is not None and val_labels is not None:
            # check if val and val_label have the same shape
            if val.shape[0] != val_labels.shape[0]:
                raise ValueError("val and val_label must have the same shape")
        MetricUtils.initializeHistory(self, val is not None)

        with tqdm(total=epochs, position=thread, desc="Epochs", colour="white") as pbar:
            for epoch in range(epochs):
                output = self.predict(inputs)

                val_output = None
                if val is not None:
                    val_output = self.predict(val)

                MetricUtils.addToHistory(self, output, input_label, val_output, val_labels)
                for callback in self.callbacks:
                    callback(self)

                if self.early_stop:
                    break

                self.curr_epoch += 1

                batches = utils.split_batches(inputs, input_label, batch_size)
                for (batch, batch_label) in batches:
                    self.optimizer.update(self, batch, batch_label)
                pbar.update(1)
                if name != '':
                    pbar.set_description(name)
                pbar.set_postfix(self.get_last())

            pbar.close()
        return self.history

    def get_last(self):
        return {name: self.history[name][-1] for name in self.history}

    def predict(self, input: np.array):
        input = input
        output = None
        for layer in self.layers:
            output = layer.feedforward(input)
            input = output
        return output

    def evaluate(self, input: np.array, input_label: np.array):
        output = self.predict(input)
        return MetricUtils.calculate('mse', target=input_label, predicted=output)
