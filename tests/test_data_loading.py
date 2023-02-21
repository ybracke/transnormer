import datasets
from transnormer.data.loader import load_dtaeval_as_dataset, load_dtaeval_all

def test_load_dtaeval_as_dataset_with_file():
    path = "tests/testdata/txt/arnima_invalide_1818-mini.txt"
    dataset = load_dtaeval_as_dataset(path)
    # document contains two sentences on either layer
    assert len(dataset["orig"]) == 2
    assert len(dataset["norm"]) == 2

def test_load_dtaeval_as_dataset_with_dir():
    path = "tests/testdata/txt/"
    dataset = load_dtaeval_as_dataset(path)
    # Sum of sentences in all documents in the folder
    assert len(dataset["orig"]) == 3
    assert len(dataset["norm"]) == 3

def test_load_dtaeval_as_dataset_with_glob():
    path = "tests/testdata/txt/*.txt"
    dataset = load_dtaeval_as_dataset(path)
    # Sum of sentences in all documents in the folder
    assert len(dataset["orig"]) == 3
    assert len(dataset["norm"]) == 3

def test_load_dtaeval_all():
    dataset_dict = load_dtaeval_all()
    assert isinstance(dataset_dict, datasets.DatasetDict)
    # print(dataset_dict)
    for part in dataset_dict:
        dataset_dict[part]
        print(part)
    #     assert len(dataset_dict[part]["norm"]) == len(part["orig"])
    #     print(len(part["norm"]))
    

