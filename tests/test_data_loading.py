import datasets
from transnormer.models.train_model import load_dtaeval_as_dataset, load_dtaeval_all

def test_load_dtaeval_as_dataset_file():
    path = "/home/bracke/data/dta/dtaeval/split-v3.1/txt/dev/spyri_heidi_1880.txt"
    dataset = load_dtaeval_as_dataset(path)
    assert len(dataset["orig"]) == 2286
    assert len(dataset["norm"]) == 2286

def test_load_dtaeval_as_dataset_dir():
    path = "/home/bracke/data/dta/dtaeval/split-v3.1/txt/dev/"
    dataset = load_dtaeval_as_dataset(path)
    assert isinstance(dataset, datasets.Dataset) 

def test_load_dtaeval_as_dataset_glob():
    path = "/home/bracke/data/dta/dtaeval/split-v3.1/txt/dev/*.txt"
    dataset = load_dtaeval_as_dataset(path)
    assert len(dataset["orig"]) == 18291
    assert len(dataset["norm"]) == 18291

def test_load_dtaeval_all():
    dataset_dict = load_dtaeval_all()
    assert isinstance(dataset_dict, datasets.DatasetDict)
    # print(dataset_dict)
    for part in dataset_dict:
        dataset_dict[part]
        print(part)
    #     assert len(dataset_dict[part]["norm"]) == len(part["orig"])
    #     print(len(part["norm"]))
    

