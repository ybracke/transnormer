# import datasets
from transnormer.data.loader import load_dtaevalxml_to_lists, load_dtaevalxml_as_dataset

# def test_load_dtaeval_as_dataset_with_file():
#     path = "tests/testdata/txt/arnima_invalide_1818-mini.txt"
#     dataset = load_dtaeval_as_dataset(path)
#     # document contains two sentences on either layer
#     assert len(dataset["orig"]) == 2
#     assert len(dataset["norm"]) == 2

# def test_load_dtaeval_as_dataset_with_dir():
#     path = "tests/testdata/txt/"
#     dataset = load_dtaeval_as_dataset(path)
#     # Sum of sentences in all documents in the folder
#     assert len(dataset["orig"]) == 3
#     assert len(dataset["norm"]) == 3

# def test_load_dtaeval_as_dataset_with_glob():
#     path = "tests/testdata/txt/*.txt"
#     dataset = load_dtaeval_as_dataset(path)
#     # Sum of sentences in all documents in the folder
#     assert len(dataset["orig"]) == 3
#     assert len(dataset["norm"]) == 3

# def test_load_dtaeval_all():
#     dataset_dict = load_dtaeval_all()
#     assert isinstance(dataset_dict, datasets.DatasetDict)


def test_load_dtaevalxml_to_lists_no_additional_filter():
    path = "tests/testdata/xml/arnima_invalide_1818-mini.xml"
    old, new = load_dtaevalxml_to_lists(path)
    # document contains 4 sentences on either layer (orig, norm)
    assert len(old) == 4
    assert len(new) == 4
    assert len(old[3]) == 15
    assert len(new[3]) == 15


def test_load_dtaevalxml_to_lists_filter_bad():
    path = "tests/testdata/xml/arnima_invalide_1818-mini.xml"
    old, new = load_dtaevalxml_to_lists(path, filter_bad=True)
    assert len(old) == 4
    assert len(new) == 4
    assert len(old[3]) == 14
    assert len(new[3]) == 14


def test_load_dtaevalxml_to_lists_filter_classes_bug():
    path = "tests/testdata/xml/arnima_invalide_1818-mini.xml"
    old, new = load_dtaevalxml_to_lists(path, filter_bad=False, filter_classes=["BUG"])
    assert len(old) == 4
    assert len(new) == 4
    assert len(old[3]) == 14
    assert len(new[3]) == 14


def test_load_dtaevalxml_as_dataset_with_file():
    path = "tests/testdata/xml/arnima_invalide_1818-mini.xml"
    dataset = load_dtaevalxml_as_dataset(path)
    # document contains 4 sentences on either layer (orig, norm)
    assert len(dataset["orig"]) == 4
    assert len(dataset["norm"]) == 4


def test_load_dtaevalxml_as_dataset_with_file_and_no_addtional_filter():
    path = "tests/testdata/xml/arnima_invalide_1818-mini.xml"
    dataset = load_dtaevalxml_as_dataset(path)
    target = "rief Basset, er sprengt sich in die Luft, rettet Euch und Euer Kind!"
    assert dataset["norm"][3] == target


def test_load_dtaevalxml_as_dataset_with_file_and_filter_classes_bug():
    path = "tests/testdata/xml/arnima_invalide_1818-mini.xml"
    dataset = load_dtaevalxml_as_dataset(path, filter_classes=["BUG"])
    target = "rief Basset, er sprengt sich in, rettet Euch und Euer Kind!"
    assert dataset["norm"][3] == target


# def test_load_dtaevalxml_all():
#     kwargs = {"filter_classes":["BUG", "FM", "GRAPH"]}
#     dataset_dict = load_dtaevalxml_all(**kwargs)
#     print(dataset_dict)
#     assert isinstance(dataset_dict, datasets.DatasetDict)
