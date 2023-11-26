import pandas as pd

__all__ = [
    "read_monk",
    "read_monk1",
    "read_monk2",
    "read_monk3"
]


def read_monk(index: int) -> tuple:
    train_set_filepath = f"exclusiveAI/datasets/monks-{index}.train"
    test_set_filepath = f"exclusiveAI/datasets/monks-{index}.test"

    # reading csvs
    train_set_df = pd.read_csv(
        train_set_filepath,
        sep=" ",
        names=["class", "a1", "a2", "a3", "a4", "a5", "a6", "id"]
    ).set_index("id")

    test_set_df = pd.read_csv(
        test_set_filepath,
        sep=" ",
        names=["class", "a1", "a2", "a3", "a4", "a5", "a6", "id"]
    ).set_index("id")

    return train_set_df, test_set_df


def read_monk1() -> tuple:
    return read_monk(1)


def read_monk2() -> tuple:
    return read_monk(2)


def read_monk3() -> tuple:
    return read_monk(3)
