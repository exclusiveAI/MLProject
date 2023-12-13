import pandas as pd
import numpy as np

__all__ = [
    "read_monk",
    "read_monk1",
    "read_monk2",
    "read_monk3"
]


def read_monk(index: int, path="exclusiveAI/datasets/") -> tuple:
    train_set_filepath = f"{path}monks-{index}.train"
    test_set_filepath = f"{path}monks-{index}.test"

    # reading csvs
    train_set_df = pd.read_csv(
        train_set_filepath,
        sep=" ",
        names=["class", "a1", "a2", "a3", "a4", "a5", "a6", "id"]
    ).set_index("id")

    train_label = np.array(train_set_df.pop("class"))
    train_set_df = np.array(train_set_df)

    test_set_df = pd.read_csv(
        test_set_filepath,
        sep=" ",
        names=["class", "a1", "a2", "a3", "a4", "a5", "a6", "id"]
    ).set_index("id")

    test_label = np.array(test_set_df.pop("class"))
    test_set_df = np.array(test_set_df)

    return train_set_df, train_label, test_set_df, test_label


def read_monk1(path="exclusiveAI/datasets/") -> tuple:
    return read_monk(1, path=path)


def read_monk2(path="exclusiveAI/datasets/") -> tuple:
    return read_monk(2, path=path)


def read_monk3(path="exclusiveAI/datasets/") -> tuple:
    return read_monk(3, path=path)
