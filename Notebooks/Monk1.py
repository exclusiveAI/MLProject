from exclusiveAI.ConfiguratorGen import ConfiguratorGen
from exclusiveAI.datasets.monk import read_monk1
from exclusiveAI.utils import one_hot_encoding
from exclusiveAI.components.Validation.HoldOut import parallel_hold_out, hold_out
import pandas as pd
import numpy as np

training_data, training_labels, test_data, test_labels = read_monk1("../exclusiveAI/datasets/")
training_data = one_hot_encoding(training_data)
test_data = one_hot_encoding(test_data)
regularizations = np.arange(0, 0.11, 0.1).tolist()
learning_rates = np.arange(0.1, 0.9, 0.01).tolist()
number_of_units = list(range(1, 5, 1))
number_of_layers = list(range(1, 3, 1))
initializers = ["uniform", "gaussian"]
momentums = np.arange(0.5, 0.99, 0.01).tolist()
momentums.insert(0, 0)
activations = ["sigmoid"]

myConfigurator = ConfiguratorGen(random=False, learning_rates=learning_rates, regularizations=regularizations,
                                 loss_function=['mse'], optimizer=['sgd'],
                                 activation_functions=activations,
                                 number_of_units=number_of_units, number_of_layers=number_of_layers,
                                 momentums=momentums, initializers=initializers,
                                 input_shapes=training_data.shape,
                                 verbose=False, nesterov=True, number_of_initializations=1,
                                 callbacks=["earlystopping"], output_activation='sigmoid', show_line=False,
                                 ).get_configs()

length = len(myConfigurator)
print("Number of configurations:", length)
buckets = 1
while length//buckets > 800000:
    buckets = buckets + 1
if buckets > 1:
    print(f"Buckets: {buckets}, Bucket size: ", length // buckets)
num_models = 2000/buckets
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
                              batch_size=batch_size, all_models=True, num_models=num_models // buckets, workers=8
                              ))

    configs = pd.DataFrame(configs)
    # Save as json
    configs.to_json('monk1_models_configurations_test2.json')
