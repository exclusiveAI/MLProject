from exclusiveAI.ConfiguratorGen import ConfiguratorGen
from exclusiveAI.datasets.mlcup import read_cup_training_dataset
from exclusiveAI.utils import train_split
from exclusiveAI.components.Validation.HoldOut import parallel_hold_out
from exclusiveAI.components.Validation.KFoldCrossValidation import validate
import pandas as pd
import numpy as np
import os
import json


file_path = "training_data_split.json"

if not os.path.exists(file_path):
    training_data, training_labels = read_cup_training_dataset("../../exclusiveAI/datasets/")
    training_data, training_labels, test_data, test_labels, train_idx, test_idx = train_split(training_data, training_labels, shuffle=True, split_size=0.1)

    # Save training and test data to a JSON file
    data_dict = {
        'training_data': training_data.tolist(),
        'training_labels': training_labels.tolist(),
        'test_data': test_data.tolist(),
        'test_labels': test_labels.tolist(),
        'train_idx': train_idx.tolist(),
        'test_idx': test_idx.tolist(),
    }

    with open(file_path, 'w') as jsonfile:
        json.dump(data_dict, jsonfile)
else:
    # Load training and test data from the JSON file
    with open(file_path, 'r') as jsonfile:
        data_dict = json.load(jsonfile)

    training_data = np.array(data_dict['training_data'])
    training_labels = np.array(data_dict['training_labels'])
    test_data = np.array(data_dict['test_data'])
    test_labels = np.array(data_dict['test_labels'])
    train_idx = np.array(data_dict['train_idx'])
    test_idx = np.array(data_dict['test_idx'])



regularizations = np.arange(0, 0.2, 0.01).tolist()
learning_rates = np.arange(0.01, 0.9, 0.01).tolist()
number_of_units = list(range(1, 6, 1))
number_of_layers = list(range(1, 4, 1))
initializers = ["uniform", "gaussian"]
momentums = np.arange(0, 0.99, 0.01).tolist()
activations = ["sigmoid", "tanh"]

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

batch_size = 16
epochs = 250
configs = []
if __name__ == '__main__':
    for run in range(4):
        for i in range(buckets):
            configs.append(
                parallel_hold_out(bucket[i], training=training_data, training_target=training_labels, epochs=epochs,
                                  batch_size=batch_size, num_models=num_models // buckets, workers=8
                                  ))

        configs = pd.DataFrame(configs)
        # Save as json
        configs.to_json(f'MLCup_1output_models_configurations_test{run}.json')
