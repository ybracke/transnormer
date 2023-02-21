from functools import wraps
import itertools
import glob
import os
import time
from typing import Generator, TextIO, Union, List, Tuple

import datasets
from lxml import etree
from nltk.tokenize.treebank import TreebankWordDetokenizer


# Helper function
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.process_time()
        result = func(*args, **kwargs)
        end = time.process_time()
        print(f"Elapsed time for `{func.__name__}`: {end - start}")
        return result

    return wrapper


def file_gen(path: str) -> Generator[TextIO, None, None]:
    """Yields file(s) from a path, where path can be file, dir or glob"""

    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as file:
            yield file
    elif os.path.isdir(path):
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), "r", encoding="utf-8") as file:
                yield file
    else:
        for filename in glob.glob(path):
            with open(filename, "r", encoding="utf-8") as file:
                yield file


def load_tsv_to_lists(
    file_obj: TextIO, keep_sentences: bool = True
) -> Union[List[List[List[str]]], List[List[str]]]:
    """
    Load corpus file from a tab-separated (CONLL-like) plain text file into a
    list.

    Each column in the input file is represented as a list inside the outer
    list. Sentences within a column are either represented as individual lists
    inside the column list or flattened so that the column list contains the
    tokens (as strings) directly .

    `keep_sentences` : if True, empty lines are interpreted as sentence breaks.
    Consecutive empty lines are ignored. Each column has the form
    `List[List[str]]`. If False the entire column content is represented as a
    single list `List[str].
    """

    line = file_obj.readline()
    # Read upto first non-empty line
    while line.isspace():
        line = file_obj.readline()
    # Number of columns in text file
    n_columns = line.strip().count("\t") + 1
    # Initial empty columns with one empty sentence inside
    columns: List[List[List[str]]] = [[[]] for i in range(n_columns)]
    # Read file
    line_cnt = 0
    sent_cnt = 0
    while line:
        # non-empty line
        if not line.isspace():
            line = line.strip()
            line_split = line.split("\t")

            # Catch/skip ill-formed lines
            if len(line_split) != n_columns:
                print(
                    f"Line {line_cnt+1} does not have length "
                    f"{n_columns} but {len(line_split)} skip line: '{line}'"
                )
            else:
                # build up sentences
                for i in range(n_columns):
                    columns[i][sent_cnt].append(line_split[i])

        # empty line
        else:
            # current sentence empty?
            # then just replace with empty sentence again
            if columns[0][sent_cnt] == []:
                for i in range(n_columns):
                    columns[i][sent_cnt] = []
            # else: move to build next sentence
            else:
                for i in range(n_columns):
                    columns[i].append([])
                sent_cnt += 1

        # Move on
        line = file_obj.readline()
        line_cnt += 1

    # optional: flatten structure
    if not keep_sentences:
        columns_flat = [list(itertools.chain(*col)) for col in columns]
        return columns_flat

    return columns


def load_dtaevalxml_to_lists(
    file: Union[str, TextIO], filter_bad: bool = False, filter_classes: List[str] = []
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Reads from a DTAEval XML file and returns two lists containing sentences as
    lists of tokens.
    """
    old = []
    new = []

    tree = etree.parse(file)
    for s in tree.iterfind("//s"):
        sent_old = []
        sent_new = []
        # skip sentences that have the @sbad attribute
        if "sbad" in s.attrib:
            continue
        # Iterate over <w> elements that are immediate childs of <s>
        # This excludes nested <w> (which are used in case of a tokenization
        # mismatch between old and new)
        for w in s.xpath("./w"):
            # Skip tokens if they meet some criterion
            tokclass = w.attrib.get("class")
            if tokclass in filter_classes:
                continue
            if "bad" in w.attrib and filter_bad:
                continue
            # Skip tokens that have no @old version (= rare errors)
            try:
                sent_old.append(w.attrib["old"])
            except KeyError:
                continue
            # Store @old as normalization if there is none (e.g. "Mur¬" -> "Mur¬")
            try:
                sent_new.append(w.attrib["new"])
            except KeyError:
                sent_new.append(w.attrib["old"])
        # append non-empty sentences
        if len(sent_old) and len(sent_new):
            old.append(sent_old)
            new.append(sent_new)
    return (old, new)


def load_dtaevalxml_as_dataset(path: str, **kwargs) -> datasets.Dataset:
    """
    Load the file(s) under `path` into a datasets.Dataset with columns "orig"
    and "norm"
    """

    # docs:
    # List <-- documents
    #     List <-- columns
    #         List <-- sentences
    #             List <-- tokens
    docs = [load_dtaevalxml_to_lists(file, **kwargs) for file in file_gen(path)]

    detokenizer = TreebankWordDetokenizer()

    # Put tokens back together into sentence strings
    docs_sent_joined = [
        [[detokenizer.detokenize(sent) for sent in column] for column in doc]
        for doc in docs
    ]

    all_sents_orig, all_sents_norm = [], []
    for doc_orig, doc_norm in docs_sent_joined:
        all_sents_orig.extend([sent for sent in doc_orig])
        all_sents_norm.extend([sent for sent in doc_norm])

    return datasets.Dataset.from_dict({"orig": all_sents_orig, "norm": all_sents_norm})


@timer
def load_dtaevalxml_all(datadir, **kwargs) -> datasets.DatasetDict:
    train_path = os.path.join(datadir, "train")
    validation_path = os.path.join(datadir, "dev")
    test_path = os.path.join(datadir, "test")

    ds = datasets.DatasetDict()
    ds["train"] = load_dtaevalxml_as_dataset(train_path, **kwargs)
    ds["validation"] = load_dtaevalxml_as_dataset(validation_path, **kwargs)
    ds["test"] = load_dtaevalxml_as_dataset(test_path, **kwargs)

    return ds


# deprecated functions

# @timer
# def load_dtaeval_all() -> datasets.DatasetDict:
#     datadir = "/home/bracke/data/dta/dtaeval/split-v3.1/txt"

#     train_path = os.path.join(datadir, "train")
#     validation_path = os.path.join(datadir, "dev")
#     test_path = os.path.join(datadir, "test")

#     ds = datasets.DatasetDict()
#     ds["train"] = load_dtaeval_as_dataset(train_path)
#     ds["validation"] = load_dtaeval_as_dataset(validation_path)
#     ds["test"] = load_dtaeval_as_dataset(test_path)

#     return ds


# def load_dtaeval_as_dataset(path: str) -> datasets.Dataset:
#     """
#     Load the file(s) under `path` into a datasets.Dataset with columns "orig"
#     and "norm"
#     """

#     docs = [load_tsv_to_lists(file, keep_sentences=True) for file in file_gen(path)]
#     docs_sent_joined = [
#         [[" ".join(sent) for sent in column] for column in doc] for doc in docs
#     ]

#     all_sents_orig, all_sents_norm = [], []
#     for doc_orig, doc_norm in docs_sent_joined:
#         all_sents_orig.extend([sent for sent in doc_orig])
#         all_sents_norm.extend([sent for sent in doc_norm])

#     return datasets.Dataset.from_dict({"orig": all_sents_orig, "norm": all_sents_norm})
