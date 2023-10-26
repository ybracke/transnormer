import random
import datasets


def sort_dataset_by_length(
    dataset: datasets.Dataset,
    column: str,
    name_length_column: str = "length",
    keep_length_column: bool = True,
    descending: bool = False,
) -> datasets.Dataset:
    """Sort a datasets.Dataset by string length of text in `column`"""
    lengths = [len(s) for s in dataset[column]]
    dataset = dataset.add_column(name_length_column, lengths)
    dataset = dataset.sort(name_length_column, reverse=descending)
    if not keep_length_column:
        dataset.remove_columns(name_length_column)

    return dataset


def filter_dataset_by_length(
    dataset: datasets.Dataset,
    max_length: int = -1,
    min_length: int = 0,
    name_length_column: str = "length",
) -> datasets.Dataset:
    """Filter a datasets.Dataset with a length column for min/max lengths"""

    # upper and lower bound
    if min_length and (max_length > -1):
        condition = (
            lambda record: record[name_length_column] >= min_length
            and record[name_length_column] <= max_length
        )  # noqa: E731
    # only upper bound
    elif max_length > -1:
        condition = (
            lambda record: record[name_length_column] <= max_length
        )  # noqa: E731
    # only lower bound
    elif min_length:
        condition = (
            lambda record: record[name_length_column] >= min_length
        )  # noqa: E731
    # no bounds, just return dataset as is
    else:
        return dataset
    return dataset.filter(condition)
