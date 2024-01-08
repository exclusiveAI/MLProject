import numpy as np
from tqdm import tqdm
import pandas as pd

__all__ = ['get_best_model']


def get_best_model(results, num_models):
    # Convert the list of tuples to a Pandas DataFrame
    df = pd.DataFrame(results, columns=['score', 'model', 'config'])

    # Order by score, then by std, and finally by model_dim
    df.sort_values(by=['score'], inplace=True)
    df.reset_index(inplace=True)
    # Select the top num_models
    df = df.head(num_models)

    return np.array(df['model']), np.array(df['config'])