from functools import wraps
import itertools
import glob
import os
import re
import time
from typing import Generator, TextIO, Union, List, Tuple, Dict, Sequence

import datasets
from lxml import etree
from nltk.tokenize.treebank import TreebankWordDetokenizer

DETOKENIZER = TreebankWordDetokenizer()


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
    file: Union[str, TextIO], keep_sentences: bool = True
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
    if isinstance(file, str):
        file_obj: TextIO = open(file, "r", encoding="utf-8")
    else:
        file_obj = file

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

    # We're done, close file
    file_obj.close()

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


@timer
def load_dtaeval_all(datadir) -> datasets.DatasetDict:
    train_path = os.path.join(datadir, "train")
    validation_path = os.path.join(datadir, "dev")
    test_path = os.path.join(datadir, "test")

    ds = datasets.DatasetDict()
    ds["train"] = load_dtaeval_as_dataset(train_path)
    ds["validation"] = load_dtaeval_as_dataset(validation_path)
    ds["test"] = load_dtaeval_as_dataset(test_path)

    return ds


def load_dtaeval_as_dataset(path: str) -> datasets.Dataset:
    """
    Load the file(s) under `path` into a datasets.Dataset with columns "orig"
    and "norm"
    """

    docs = [load_tsv_to_lists(file, keep_sentences=True) for file in file_gen(path)]
    docs_sent_joined = [
        [[" ".join(sent) for sent in column] for column in doc] for doc in docs
    ]

    all_sents_orig, all_sents_norm = [], []
    for doc_orig, doc_norm in docs_sent_joined:
        all_sents_orig.extend([sent for sent in doc_orig])
        all_sents_norm.extend([sent for sent in doc_norm])

    return datasets.Dataset.from_dict({"orig": all_sents_orig, "norm": all_sents_norm})


def filepath_gen(path: str) -> Generator[str, None, None]:
    """Yields filepath(s) from a path, where path can be file, dir or glob"""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for filename in os.listdir(path):
            yield os.path.join(path, filename)
    else:
        for filename in glob.glob(path):
            yield filename


def detokenize_doc(doc: Sequence[List[List[str]]]) -> List[List[str]]:
    """Performs detokenization (List[str] -> str) on all sentences in a multi-column doc"""
    return [[DETOKENIZER.detokenize(sent) for sent in column] for column in doc]


def extract_year(input_string: str) -> str:
    """
    Extracts the first four consecutive digits from a string
    (or an empty string if there is no match)

    In our scenario, this should get us the publication year from a filename
    """
    match = re.search(r"\d\d\d\d", input_string)
    if match:
        return match.group(0)
    else:
        return ""


def _find_split(input_string: str) -> str:
    """
    Return train|validation|test|{empty string} depending on the input

    In our scenario, the input should be a file or directory path
    """
    if "train" in input_string:
        split = "train"
    elif ("dev" in input_string) or ("validation" in input_string):
        split = "validation"
    elif "test" in input_string:
        split = "test"
    # dataset has not been split
    else:
        split = ""
    return split


def load_data(
    paths: List[str],
) -> Generator[Tuple[str, str, Dict[str, List[str]]], None, None]:
    """
    Generator that returns the name of a dataset, the split, and the actual data

    `data` is a dict as returned by a `read_*` function, which looks like this:
    { "orig" : [...], "norm" : [...], + optional metadata }

    Function is inspired by:
    `https://github.com/zentrum-lexikographie/eval-de-pos/blob/main/src/loader.py`
    """

    # default outputs
    dname = ""
    split = ""
    o: Dict[str, List[str]] = {"orig": [], "norm": []}

    for path in paths:
        # Call read_ function depending on dataset
        if "dtaeval" in path:
            # TODO|s
            # - Is the match check for the dataset path/name okay like this?
            # - Same question for _find_split
            # - Where/how do the filter_kwargs get passed for filtering certain XML elements
            filter_kwargs: Dict[str, Union[str, List[str]]] = {}
            o = read_dtaeval_raw(path, metadata=True, **filter_kwargs)
            dname = "dtaeval"
            split = _find_split(path)

        elif "ridges/bollmann-split" in path:
            o = read_ridges_raw(path)
            dname = "ridges_bollmann"
            split = _find_split(path)

        elif "germanc" in path:
            pass
            # o = read_germanc_raw(path, metadata=True)
            # dname = "germanc_gs"
            # split = _find_split(path)

        elif "deu_news_2020" in path:
            o = read_leipzig_raw(path)
            dname = "deu_news_2020"
            split = _find_split(path)

        yield (dname, split, o)


def read_dtaeval_raw(
    path: str, metadata=False, **filter_kwargs
) -> Dict[str, List[str]]:
    """
    Read in a part of DTA EvalCorpus (XML version) and return it as a dict

    Returns: {"orig" : [...], "norm" : [...], + optional metadata }
    """
    all_sents_orig, all_sents_norm = [], []
    if metadata:
        all_years, all_docs = [], []
    for docpath in filepath_gen(path):
        # Load document into a list of tokenized sentences
        # The two elements in the outermost list are orig and norm columns
        doc_tok = load_dtaevalxml_to_lists(docpath, **filter_kwargs)
        # Sentences: List[str] -> str
        doc = detokenize_doc(doc_tok)
        # Collect all sentences in list
        all_sents_orig.extend([sent for sent in doc[0]])
        all_sents_norm.extend([sent for sent in doc[1]])
        if metadata:
            basename = os.path.splitext(os.path.basename(docpath))[0]
            year = extract_year(basename)
            all_years.extend([year for i in range(len(doc[0]))])
            all_docs.extend([basename for i in range(len(doc[0]))])

    if metadata:
        return {
            "orig": all_sents_orig,
            "norm": all_sents_norm,
            "year": all_years,
            "document": all_docs,
        }

    return {"orig": all_sents_orig, "norm": all_sents_norm}


def read_ridges_raw(path: str) -> Dict[str, List[str]]:
    """
    Read in a part of the RIDGES Corpus (plain text version) and return it as a dict

    This is for the plain text (tsv) version of RIDGES corpus provided by Marcel Bollmann:
    https://github.com/coastalcph/histnorm/tree/master/datasets/historical/german
    There is no metadata available for this corpus version.

    Returns: {"orig" : [...], "norm" : [...]}
    """
    all_sents_orig, all_sents_norm = [], []
    for docpath in filepath_gen(path):
        # Load document into a list of tokenized sentences
        # The two elements in the outermost list are orig and norm columns
        doc_tok = load_tsv_to_lists(docpath, keep_sentences=True)
        # Sentences are converted from List[str] to str
        doc = detokenize_doc(doc_tok)
        # Collect all sentences in list
        all_sents_orig.extend([sent for sent in doc[0]])
        all_sents_norm.extend([sent for sent in doc[1]])

    return {"orig": all_sents_orig, "norm": all_sents_norm}


def read_leipzig_raw(path: str) -> Dict[str, List[str]]:
    """
    Read in Leipzig Corpora Collection file(s) and return it as a dict

    Files are plain text and can be downloaded here:
    https://wortschatz.uni-leipzig.de/de/download
    There is no metadata available for this corpus.
    The texts are already in standard modern German, so "norm" is
    just a copy of "orig".

    Returns: {"orig" : [...], "norm" : [...]}
    """
    all_sents = []
    for docpath in filepath_gen(path):
        # Load document
        with open(docpath, "r", encoding="utf-8") as f:
            doc = f.readlines()
        # Collect all sentences in list
        all_sents.extend(doc)

    return {"orig": all_sents, "norm": all_sents}


# def read_germanc_raw(path: str) -> Dict[str, List[str]]:
#     return


def merge_datasets(
    dsets: Union[List[datasets.Dataset], Dict[datasets.Dataset, int]], seed: int
) -> datasets.Dataset:
    """
    Merge multiple datasets into a single dataset with optional resampling.

    If a list is passed as the argument, the datasets are concatenated. If a
    dict is passed the keys are taken to be datasets and the values are taken to
    be the desired number of samples for that dataset. Before concatenating
    datasets are pruned to that number of samples.
    If one of the passed values exceeds the `.num_rows` of the respective dataset,
    n is set to equal `.num_rows`.
    """
    if isinstance(dsets, list):
        merged_dataset = datasets.concatenate_datasets(dsets)
    elif isinstance(dsets, dict):
        dsets_resampled = []
        for ds, n in dsets.items():
            # Catch `n`s that exceed number of rows
            if n > ds.num_rows:
                n = ds.num_rows
            # Shuffle and resample
            dsets_resampled.append(ds.shuffle(seed=seed).select(range(n)))
        merged_dataset = datasets.concatenate_datasets(dsets_resampled)
    else:
        raise TypeError("Argument `dsets` must be of type list or dict.")
    return merged_dataset
