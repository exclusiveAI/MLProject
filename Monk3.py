from exclusiveAI.ConfiguratorGen import ConfiguratorGen
from exclusiveAI.datasets.monk import read_monk3
from exclusiveAI.utils import one_hot_encoding
from exclusiveAI.components.Validation.HoldOut import parallel_hold_out, hold_out
import pandas as pd
import numpy as np

training_data, training_labels, test_data, test_labels = read_monk3("exclusiveAI/datasets/")
training_data = one_hot_encoding(training_data)
test_data = one_hot_encoding(test_data)
regularizations = np.arange(0, 0.001, 0.0001)
learning_rates = np.arange(0.01, 0.5, 0.01)
number_of_units = range(1, 4, 1)
number_of_layers = range(1, 2, 1)
initializers = ["uniform", "gaussian"]
momentums = np.arange(0, 0.999, 0.001)
activations = ["sigmoid"]

myConfigurator = ConfiguratorGen(random=False, learning_rates=learning_rates, regularizations=regularizations,
                                 loss_function=['mse'], optimizer=['sgd'],
                                 activation_functions=activations,
                                 number_of_units=number_of_units, number_of_layers=number_of_layers,
                                 momentums=momentums, initializers=initializers,
                                 input_shapes=training_data.shape,
                                 verbose=False, nesterov=True, number_of_initializations=1,
                                 callbacks=["earlystopping"], output_activation='sigmoid', show_line=True,
                                 ).get_configs()

length = len(myConfigurator)
buckets = 100

bucket = {}
for i in range(buckets):
    bucket[i] = myConfigurator[i * length // buckets:(i + 1) * length // buckets if i + 1 < buckets else length]

batch_size = 32
epochs = 200
configs = []
if __name__ == '__main__':
    for i in range(buckets):
        configs.append(
            parallel_hold_out(bucket[i], training=training_data, training_target=training_labels, epochs=epochs,
                              batch_size=batch_size, all_models=True, num_models=500 // buckets))

    configs = pd.DataFrame(configs)
    # Save as json
    configs.to_json('monk3_models_configurations.json')
