import numpy as np
from exclusiveAI.components.LossFunctions import LossFunction
from exclusiveAI.components.Optimizers import Optimizer
from exclusiveAI.components.Metrics import MetricUtils
from exclusiveAI import utils
from tqdm import tqdm


class neural_network:
    def __init__(self,
                 layers: list,
                 loss: LossFunction,
                 optimizer: Optimizer,
                 learning_rate: float,
                 callbacks: list,
                 metrics: [],
                 verbose: bool = False,
                 ):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.loss_function = loss
        self.layers = layers
        self.verbose = verbose
        self.early_stop = False

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
              epochs=100, batch_size=32):
        # check if both val and val_label are provided
        if val is not None and val_labels is not None:
            # check if val and val_label have the same shape
            if val.shape != val_labels.shape:
                raise ValueError("val and val_label must have the same shape")
        MetricUtils.initializeHistory(self, val is not None)

        with tqdm(total=epochs, desc="Epochs") as pbar:
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
                # if self.verbose:
                #     print(f"Epoch {self.curr_epoch}/{epochs}")

                batches = utils.split_batches(inputs, input_label, batch_size)
                for (batch, batch_label) in batches:
                    self.optimizer.update(self, batch, batch_label)
                pbar.update(1)
                for metric in self.history:
                    pbar.write(f"{metric}: {self.history[metric][-1]}")
                # pbar.set_postfix(loss=self.history[-1]['loss'])
            return self.history

    def predict(self, input: np.array):
        input = input
        output = None
        for layer in self.layers:
            output = layer.feedforward(input)
            input = output
        return output
