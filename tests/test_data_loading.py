import datasets
from transnormer.data import loader

# def test_load_dtaeval_as_dataset_with_file():
#     path = "tests/testdata/dtaeval/txt/arnima_invalide_1818-mini.txt"
#     dataset = loader.load_dtaeval_as_dataset(path)
#     # document contains two sentences on either layer
#     assert len(dataset["orig"]) == 2
#     assert len(dataset["norm"]) == 2

# def test_load_dtaeval_as_dataset_with_dir():
#     path = "tests/testdata/dtaeval/txt/"
#     dataset = loader.load_dtaeval_as_dataset(path)
#     # Sum of sentences in all documents in the folder
#     assert len(dataset["orig"]) == 3
#     assert len(dataset["norm"]) == 3

# def test_load_dtaeval_as_dataset_with_glob():
#     path = "tests/testdata/dtaeval/txt/*.txt"
#     dataset = loader.load_dtaeval_as_dataset(path)
#     # Sum of sentences in all documents in the folder
#     assert len(dataset["orig"]) == 3
#     assert len(dataset["norm"]) == 3

# def test_load_dtaeval_all():
#     dataset_dict = loader.load_dtaeval_all()
#     assert isinstance(dataset_dict, datasets.DatasetDict)


def test_load_tsv_to_lists_filename():
    path = "tests/testdata/dtaeval/txt/arnima_invalide_1818-head10.txt"
    target = [
        [
            [
                "Auch",
                "im",
                "ſüdlichen",
                "Frankreich",
                "iſt",
                "es",
                "nicht",
                "immer",
                "warm",
                ",",
            ]
        ],
        [
            [
                "Auch",
                "im",
                "südlichen",
                "Frankreich",
                "ist",
                "es",
                "nicht",
                "immer",
                "warm",
                ",",
            ]
        ],
    ]
    doc = loader.load_tsv_to_lists(path)
    assert doc == target
    pass


def test_load_tsv_to_lists_opened_file():
    path = "tests/testdata/dtaeval/txt/arnima_invalide_1818-head10.txt"
    target = [
        [
            [
                "Auch",
                "im",
                "ſüdlichen",
                "Frankreich",
                "iſt",
                "es",
                "nicht",
                "immer",
                "warm",
                ",",
            ]
        ],
        [
            [
                "Auch",
                "im",
                "südlichen",
                "Frankreich",
                "ist",
                "es",
                "nicht",
                "immer",
                "warm",
                ",",
            ]
        ],
    ]
    with open(path, "r", encoding="utf-8") as f:
        doc = loader.load_tsv_to_lists(f)
    assert doc == target


def test_load_dtaevalxml_to_lists_no_additional_filter():
    path = "tests/testdata/dtaeval/xml/arnima_invalide_1818-mini.xml"
    old, new = loader.load_dtaevalxml_to_lists(path)
    # document contains 4 sentences on either layer (orig, norm)
    assert len(old) == 4
    assert len(new) == 4
    assert len(old[3]) == 15
    assert len(new[3]) == 15


def test_load_dtaevalxml_to_lists_filter_bad():
    path = "tests/testdata/dtaeval/xml/arnima_invalide_1818-mini.xml"
    old, new = loader.load_dtaevalxml_to_lists(path, filter_bad=True)
    assert len(old) == 4
    assert len(new) == 4
    assert len(old[3]) == 14
    assert len(new[3]) == 14


def test_load_dtaevalxml_to_lists_filter_classes_bug():
    path = "tests/testdata/dtaeval/xml/arnima_invalide_1818-mini.xml"
    old, new = loader.load_dtaevalxml_to_lists(
        path, filter_bad=False, filter_classes=["BUG"]
    )
    assert len(old) == 4
    assert len(new) == 4
    assert len(old[3]) == 14
    assert len(new[3]) == 14


def test_load_dtaevalxml_as_dataset_with_file():
    path = "tests/testdata/dtaeval/xml/arnima_invalide_1818-mini.xml"
    dataset = loader.load_dtaevalxml_as_dataset(path)
    # document contains 4 sentences on either layer (orig, norm)
    assert len(dataset["orig"]) == 4
    assert len(dataset["norm"]) == 4


def test_load_dtaevalxml_as_dataset_with_file_and_no_addtional_filter():
    path = "tests/testdata/dtaeval/xml/arnima_invalide_1818-mini.xml"
    dataset = loader.load_dtaevalxml_as_dataset(path)
    target = "rief Basset, er sprengt sich in die Luft, rettet Euch und Euer Kind!"
    assert dataset["norm"][3] == target


def test_load_dtaevalxml_as_dataset_with_file_and_filter_classes_bug():
    path = "tests/testdata/dtaeval/xml/arnima_invalide_1818-mini.xml"
    dataset = loader.load_dtaevalxml_as_dataset(path, filter_classes=["BUG"])
    target = "rief Basset, er sprengt sich in, rettet Euch und Euer Kind!"
    assert dataset["norm"][3] == target


# def test_load_dtaevalxml_all():
#     kwargs = {"filter_classes":["BUG", "FM", "GRAPH"]}
#     dataset_dict = loader.load_dtaevalxml_all(**kwargs)
#     print(dataset_dict)
#     assert isinstance(dataset_dict, datasets.DatasetDict)


def test_extract_year_normal():
    str = "/dev/brentano_kasperl_1838.xml"
    assert loader.extract_year(str) == "1838"


def test_extract_year_noyear():
    str = "/dev/brentano_kasperl.xml"
    assert loader.extract_year(str) == ""


def test_extract_year_twoyears():
    str = "/dev/brentano_kasperl_1838-1840.xml"
    assert loader.extract_year(str) == "1838"


def test_read_dtaeval_raw_no_metadata():
    path = "tests/testdata/dtaeval/xml/arnima_invalide_1818-head10.xml"
    data = loader.read_dtaeval_raw(path)
    target_data = {
        "orig": ["Graf Dürande, der gute alte Kommandant von Marſeille,"],
        "norm": ["Graf Dürande, der gute alte Kommandant von Marseille,"],
    }
    assert data == target_data


def test_read_dtaeval_raw_with_metadata():
    path = "tests/testdata/dtaeval/xml/arnima_invalide_1818-head10.xml"
    data = loader.read_dtaeval_raw(path, metadata=True)
    target_data = {
        "orig": ["Graf Dürande, der gute alte Kommandant von Marſeille,"],
        "norm": ["Graf Dürande, der gute alte Kommandant von Marseille,"],
        "year": ["1818"],
        "document": ["arnima_invalide_1818-head10"],
    }
    assert data == target_data


def test_read_ridges_raw():
    path = "tests/testdata/ridges/ridges.train.head-10.txt"
    data = loader.read_ridges_raw(path)
    target_data = {
        "orig": [
            "AN diſem fünfften ſtucke des puchs ſoͤll wir ſagen von den kreüteren vnd des erſten in eyner gemein"
        ],
        "norm": [
            "An diesem fünften Stück des Buchs sollen wir sagen von den Kräutern und des ersten in einer Gemein"
        ],
    }
    assert data == target_data


def test_read_leipzig_raw():
    path = "/home/bracke/code/transnormer/data/raw/leipzig-corpora/deu_news_2020_1M-sentences.txt"
    d = loader.read_leipzig_raw(path)
    print(len(d["orig"]))


def test_load_data():
    paths = ["tests/testdata/dtaeval/xml"]
    for i, (dname, split, o) in enumerate(loader.load_data(paths)):
        if i == 0:
            assert dname == "dtaeval"
            assert split == "test"
            assert all(map(lambda x: x in o, ["orig", "norm", "year", "document"]))
            assert len(o["orig"]) > 0
            assert len(o["orig"]) == len(o["norm"])


def test_merge_datasets():
    seed = 42
    path_ds1 = "tests/testdata/dtaeval/xml/arnima_invalide_1818-head10.xml"
    path_ds2 = "tests/testdata/ridges/ridges.train.head-10.txt"
    ds1 = datasets.Dataset.from_dict(loader.read_dtaeval_raw(path_ds1))
    ds2 = datasets.Dataset.from_dict(loader.read_ridges_raw(path_ds2))
    # Create a version of ds1 with 4 texts (by concatenating it with itself)
    ds1 = datasets.concatenate_datasets([ds1] * 4)
    # Similar for ds2
    ds2 = datasets.concatenate_datasets([ds2] * 4)
    # Get a proportion of 4:1 in the final set
    ds = loader.merge_datasets({ds1: 4, ds2: 1}, seed=seed)
    assert ds.num_rows == 5
    # This still gets a proportion of 4:1 in the final set
    ds = loader.merge_datasets({ds1: 1000, ds2: 1}, seed=seed)
    assert ds.num_rows == 5
    # This still gets a proportion of 1:1 (or whatever the original proportion
    # of the datasets is) in the final set
    ds = loader.merge_datasets([ds1, ds2], seed=seed)
    assert ds.num_rows == 8
