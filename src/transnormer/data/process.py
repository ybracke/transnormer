import datasets


def sort_dataset_by_length(
    dataset: datasets.Dataset,
    column: str,
    keep_length_column: bool = True,
    name_length_column: str = "length",
    add_original_index: bool = True,
    name_index_column: str = "#",
    descending: bool = False,
    use_bytelength: bool = False,
) -> datasets.Dataset:
    """Sort a datasets.Dataset by string length of text in `column`"""
    if use_bytelength:
        lengths = [len(bytes(s, "utf8")) for s in dataset[column]]
    else:
        lengths = [len(s) for s in dataset[column]]
    dataset = dataset.add_column(name_length_column, lengths)
    if add_original_index:
        indexes = [i for i in range(len(dataset))]
        dataset = dataset.add_column(name_index_column, indexes)
    dataset = dataset.sort(name_length_column, reverse=descending)
    if not keep_length_column:
        dataset = dataset.remove_columns(name_length_column)

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
