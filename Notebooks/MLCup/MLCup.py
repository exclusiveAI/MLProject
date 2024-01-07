from exclusiveAI.ConfiguratorGen import ConfiguratorGen
from exclusiveAI.datasets.mlcup import read_cup_training_dataset
from exclusiveAI.utils import train_split
from exclusiveAI.components.Validation.KFoldCrossValidation import validate
import pandas as pd
import numpy as np
import os
import json

file_path = "Notebooks/MLCup/Data/training_data_split.json"

if not os.path.exists(file_path):
    training_data, training_labels = read_cup_training_dataset("exclusiveAI/datasets/")
    training_data, training_labels, test_data, test_labels, train_idx, test_idx = train_split(training_data,
                                                                                              training_labels,
                                                                                              shuffle=True,
                                                                                              split_size=0.25)

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

regularizations = [1e-8, 1e-7, 1e-6]
learning_rates = np.arange(0.02, 0.06, 0.01)
number_of_units = [10, 15, 20]
number_of_layers = [2, 3]
initializers = ["uniform", "gaussian"]
momentums = np.arange(0.1, 0.6, 0.1).tolist()
momentums = [round(value, 2) for value in momentums]
activations = ["sigmoid", "tanh"]
if __name__ == '__main__':

    myConfigurator = ConfiguratorGen(random=False, learning_rates=learning_rates, regularizations=regularizations,
                                     loss_function=['mse'], optimizer=['sgd'],
                                     activation_functions=activations,
                                     number_of_units=number_of_units, number_of_layers=number_of_layers,
                                     momentums=momentums, initializers=initializers,
                                     input_shapes=training_data.shape,
                                     verbose=False, nesterov=True, outputs=3,
                                     callbacks=["earlystopping"], output_activation='linear', show_line=False,
                                     ).get_configs()

    length = len(myConfigurator)
    print("Number of configurations:", length)
    buckets = 1
    while length // buckets > 800000:
        buckets = buckets + 1
    if buckets > 1:
        print(f"Buckets: {buckets}, Bucket size: ", length // buckets)
    num_models = 2000
    bucket = {}
    for i in range(buckets):
        bucket[i] = myConfigurator[i * length // buckets:(i + 1) * length // buckets if i + 1 < buckets else length]

    batch_size = 200
    epochs = 100
    configs = []
    for i in range(buckets):
        configs.append(
            validate(bucket[i], x=training_data, y_true=training_labels, metric='val_mse', max_configs=num_models,
                     regression=True,
                     n_splits=4, epochs=epochs, batch_size=batch_size, eps=1e-2, workers=8))
        if buckets > 1:
            configs = pd.DataFrame(configs)
            # Save as json
            configs.to_json(f'MLCup_1output_models_configurations_test_1_bucket_{i}.json')
    if buckets == 1:
        configs = pd.DataFrame(configs)
        # Save as json
        configs.to_json(f'MLCup_1output_models_configurations_test_1.json')
