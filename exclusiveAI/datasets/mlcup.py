import pandas as pd
import numpy as np
from exclusiveAI.utils import train_split

__all__ = [
    "read_cup_training_dataset",
    "train_val_test_split",
    "read_cup_test_dataset"
]


def read_cup_training_dataset(path="exclusiveAI/datasets"):
    training_set_filepath = f"{path}/ML-CUP23-TR.csv"

    training_set_df = pd.read_csv(
        training_set_filepath,
        sep=",",
        dtype='float64',
        names=["id", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "TargetX", "TargetY", "TargetZ"],
        skiprows=7,
    ).set_index("id")

    # Extract and remove the last three columns as labels
    training_set_labels = np.array(training_set_df[["TargetX", "TargetY", "TargetZ"]].copy())
    training_set_df.drop(["TargetX", "TargetY", "TargetZ"], axis=1, inplace=True)

    training_set_df = np.array(training_set_df)

    return training_set_df, training_set_labels


def train_val_test_split(training_set_df, training_set_labels, train_size=0.7, test_size=0.33):
    train_set, train_labels, tmp_set, tmp_labels, train_idx, tmp_idx = train_split(training_set_df,
                                                                                   training_set_labels,
                                                                                   split_size=1 - train_size)
    val_set, val_labels, test_set, test_labels, val_idx, test_idx = train_split(tmp_set,
                                                                                tmp_labels,
                                                                                split_size=test_size)

    return (train_set, train_labels,
            val_set, val_labels, test_set,
            test_labels, train_idx,
            tmp_idx[val_idx], tmp_idx[test_idx])


def read_cup_test_dataset(path='exclusiveAI/datasets'):
    test_set_filepath = f"{path}/ML-CUP23-TS.csv"

    test_set_df = pd.read_csv(
        test_set_filepath,
        sep=",",
        dtype='float64',
        names=["id", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"],
        skiprows=7,
    ).set_index("id")

    test_set_df = np.array(test_set_df)

    return test_set_df
