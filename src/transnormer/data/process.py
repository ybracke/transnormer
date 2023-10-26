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


def shuffle_dataset_in_chunks(
    dataset: datasets.Dataset, chunk_size: int
) -> datasets.Dataset:
    """
    Shuffle a datasets.Dataset in chunks of size `chunk_size`

    Example: This chops the dataset in chunks of size 32 and randomly orders the chunks.
    Note: The internal order of the chunks is not altered.
    """

    num_chunks = len(dataset) // chunk_size
    if len(dataset) % chunk_size:
        num_chunks += 1
    chunks = [dataset.shard(num_chunks, i, contiguous=True) for i in range(num_chunks)]
    random.shuffle(chunks)
    shuffled_ds = datasets.concatenate_datasets(chunks)

    return shuffled_ds


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
