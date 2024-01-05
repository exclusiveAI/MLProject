import numpy as np
from tqdm import tqdm
import pandas as pd

__all__ = ['get_best_model']


def get_best_model(results, eps, num_models):
    best_models = []

    for score, std, model, config in tqdm(results, desc="Processing models", unit="model", leave=False):
        for i, ith_model in enumerate(best_models):
            if score < ith_model[1]:# or (abs(score - ith_model[1]) < eps and compare_model_dim(model, ith_model[3]) < 0):
                best_models.insert(i, (std, score, model_dim(model), model, config))
                break

        if not best_models:
            best_models.insert(0, (std, score, model_dim(model), model, config))

    # Convert the list of tuples to a Pandas DataFrame
    df = pd.DataFrame(best_models, columns=['std', 'score', 'model_dim', 'model', 'config'])

    # Order by score, then by std, and finally by model_dim
    df.sort_values(by=['score', 'model_dim', 'std'], inplace=True)
    # Select the top num_models
    df = df.head(num_models)

    return np.array(df['model']), np.array(df['config'])

# def get_best_model(results, metric, eps, num_models):
#     best_std = []
#     best_score = []
#     best_models = []
#     best_configs = []
#
#     # Wrap the loop with tqdm to add a progress bar
#     for score, std, model, config in tqdm(results, desc="Processing models", unit="model", leave=False):
#         if not best_models:
#             best_std.append(std)
#             best_score.append(score)
#             best_models.append(model)
#             best_configs.append(config)
#         else:
#             for i, ith_model in enumerate(best_models):
#                 delta_metrics = model.get_last()[metric] - ith_model.get_last()[metric]
#                 model_layers_delta = compare_model_dim(model, ith_model)
#                 if delta_metrics < 0 or (abs(delta_metrics) < eps and (model_layers_delta < 0)):
#                     best_std.insert(i, std)
#                     best_score.insert(i, score)
#                     best_models.insert(i, model)
#                     best_configs.insert(i, config)
#                     break
#             if len(best_models) > num_models:
#                 best_std = best_std[:num_models]
#                 best_score = best_score[:num_models]
#                 best_models = best_models[:num_models]
#                 best_configs = best_configs[:num_models]
#
#     return best_models, best_configs


def compare_model_dim(model1, model2):
    if len(model1.layers) < len(model2.layers):
        return -1
    if len(model1.layers) == len(model2.layers):
        counter = 0
        for layer_model1, layer_model2 in zip(model1.layers, model2.layers):
            if layer_model1.units < layer_model2.units:
                counter += 1

        return counter - len(model1.layers) if counter > 0 else counter
    return 0


def model_dim(model):
    layers = [layer_model.units for i, layer_model in enumerate(model.layers)]
    layers_sum = np.sum(layers)
    return len(model.layers) + layers_sum
