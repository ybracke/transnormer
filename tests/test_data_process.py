import datasets
import torch

from transnormer.data import process

# Fix seeds for reproducibilty
SEED = 42
torch.manual_seed(SEED)

DATA = {
    "input_ids": [
        torch.randint(0, 128, (50,)),
        torch.randint(0, 128, (20,)),
        torch.randint(0, 128, (90,)),
        torch.randint(0, 128, (80,)),
        torch.randint(0, 128, (10,)),
        torch.randint(0, 128, (40,)),
        torch.randint(0, 128, (100,)),
        torch.randint(0, 128, (70,)),
        torch.randint(0, 128, (30,)),
        torch.randint(0, 128, (60,)),
    ]
}

DATASET = datasets.Dataset.from_dict(DATA)
lengths = [len(s) for s in DATASET["input_ids"]]
DATASET_WITH_LENGTHS = DATASET.add_column("length", lengths)


def test_sort_dataset_by_length_asc() -> None:
    sorted_ds = process.sort_dataset_by_length(DATASET, column="input_ids")
    sorted_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for i in range(10):
        assert sorted_ds[i]["length"] == sorted_lengths[i]


def test_sort_dataset_by_length_desc() -> None:
    sorted_ds = process.sort_dataset_by_length(
        DATASET, column="input_ids", descending=True
    )
    sorted_lengths = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    for i in range(10):
        assert sorted_ds[i]["length"] == sorted_lengths[i]


def test_filter_dataset_by_length_max_length_70() -> None:
    filtered_ds = process.filter_dataset_by_length(DATASET_WITH_LENGTHS, 70)
    assert len(filtered_ds) == 7


def test_filter_dataset_by_length_max_length_0() -> None:
    filtered_ds = process.filter_dataset_by_length(DATASET_WITH_LENGTHS, 0)
    assert len(filtered_ds) == 0


def test_filter_dataset_by_length_min_length_20() -> None:
    filtered_ds = process.filter_dataset_by_length(DATASET_WITH_LENGTHS, min_length=20)
    assert len(filtered_ds) == 9


def test_filter_dataset_by_length_min_length_20_max_length_70() -> None:
    filtered_ds = process.filter_dataset_by_length(
        DATASET_WITH_LENGTHS, 70, min_length=20
    )
    assert len(filtered_ds) == 6


def test_filter_dataset_by_length_max_length_negative() -> None:
    filtered_ds = process.filter_dataset_by_length(DATASET_WITH_LENGTHS, -10)
    assert len(filtered_ds) == 10


def test_filter_dataset_by_length_min_length_negative() -> None:
    filtered_ds = process.filter_dataset_by_length(DATASET_WITH_LENGTHS, min_length=-10)
    assert len(filtered_ds) == 10


def test_filter_dataset_by_length_min_and_max_length_negative() -> None:
    filtered_ds = process.filter_dataset_by_length(
        DATASET_WITH_LENGTHS, -10, min_length=-10
    )
    assert len(filtered_ds) == 10
