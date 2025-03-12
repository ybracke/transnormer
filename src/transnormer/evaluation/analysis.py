import random
import re
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from transnormer.evaluation import tokenise
from transnormer.evaluation.align_levenshtein import align


def keep_error_sents(df: pd.DataFrame) -> pd.DataFrame:
    """In dataframe keep only the sentences with 'score' < 1.0"""
    return df[df["score"] < 1.0]


def get_alignments(df: pd.DataFrame, cache_file: Optional[str] = None) -> pd.DataFrame:
    """
    Adds additional columns '{orig,norm,pred}_tok', 'alignm_orig2norm', 'alignm_orig2pred' to a dataframe with columns 'orig', 'norm', 'pred'

    cache_file can be used to load and dump (pre-)computed alignments
    """

    # Tokenize for alignment
    orig_tok = [tokenise.basic_tokenise(sent) for sent in df["orig"]]
    norm_tok = [tokenise.basic_tokenise(sent) for sent in df["norm"]]
    pred_tok = [tokenise.basic_tokenise(sent) for sent in df["pred"]]

    # Compute alignments (use cache)
    alignments_orig2norm = align(orig_tok, norm_tok, cache_file=cache_file)
    alignments_orig2pred = align(orig_tok, pred_tok, cache_file=cache_file)

    # Add multiple columns using dictionary assignment
    new_data = {
        "orig_tok": orig_tok,
        "norm_tok": norm_tok,
        "pred_tok": pred_tok,
        "alignm_orig2norm": alignments_orig2norm,
        "alignm_orig2pred": alignments_orig2pred,
    }
    df = df.assign(**new_data)
    return df


def make_error_tokens_df(sents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe that looks like this


    | sent_id | orig_tok | norm_tok | pred_tok | alignm_orig2norm | alignm_orig2pred |
    |---------|----------|----------|----------|------------------|------------------|
    |         |          |          |          |                  |                  |


    and creates a dataframe that looks like this:

    | id | orig | gold | pred | sent_id |
    |----|------|------|------|---------|
    |    |      |      |      |         |
    """
    # List of dictionaries for output dataframe
    records = []

    # Iterate over sents_df
    for index, row in sents_df.iterrows():
        # For each sent: Iterate over zipped alignments
        # correct_sent, total_sent = 0, 0
        # Remove spaces
        alignments_orig2norm = [
            (orig1, gold)
            for (orig1, gold, _) in row["alignm_orig2norm"]
            if (orig1, gold) != ("", "")
        ]
        alignments_orig2pred = [
            (orig1, pred)
            for (orig1, pred, _) in row["alignm_orig2pred"]
            if (orig1, pred) != ("", "")
        ]

        for (orig1, gold), (orig2, pred) in zip(
            alignments_orig2norm, alignments_orig2pred
        ):
            assert orig1 == orig2, (orig1, orig2)
            # skip spaces
            if orig1 == "":
                continue
            # if gold and pred don't match: add a record
            if gold != pred:
                record = {"orig": orig1, "gold": gold, "pred": pred, "sent_id": index}
                records.append(record)

    return pd.DataFrame.from_records(records)


def make_error_type_df(error_tokens_df: pd.DataFrame):
    # Go over error_tokens_df: collect sent_ids in a list for each
    # triple (orig, norm, pred)
    errortriples2sent_ids: Dict[Tuple[str, str, str], List[int]] = {}
    for index, row in error_tokens_df.iterrows():
        triple = (row["orig"], row["gold"], row["pred"])
        if triple not in errortriples2sent_ids:
            errortriples2sent_ids[triple] = [row["sent_id"]]
        else:
            errortriples2sent_ids[triple].append(row["sent_id"])

    # Create a dataframe from triples2sent_ids
    records = [
        {"orig": t[0], "gold": t[1], "pred": t[2], "sent_ids": sorted(sent_ids)}
        for t, sent_ids in errortriples2sent_ids.items()
    ]
    df = pd.DataFrame.from_records(records)

    # Add a column for count (= length of sent_ids)
    df = df.assign(count=[len(row["sent_ids"]) for _, row in df.iterrows()])

    return df


def add_invoc_column(df: pd.DataFrame, vocab: Iterable[str]) -> pd.DataFrame:
    """Adds a column to error type df that specifies whether orig is invoc/oov"""
    df = df.assign(invoc=[row["orig"] in vocab for _, row in df.iterrows()])
    return df


def add_error_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds three columns (space_only, case_only, e-related, other) to specify the type of error
    """
    drop_spacing = str.maketrans("", "", "▁░")
    new_data: Dict[str, List[bool]] = {
        "space_only": [],
        "case_only": [],
        "e-related": [],
        "other": [],
    }

    for _, row in df.iterrows():
        gold, pred = row["gold"], row["pred"]
        space_only = gold.translate(drop_spacing) == pred.translate(drop_spacing)
        case_only = gold.lower() == pred.lower()
        # ~ gehet -> geht
        # Dative -e; Baume -> Baum
        e_related = (
            re.sub(r"(^.+)e(.$)", r"\1\2", gold) == re.sub(r"(^.+)e(.$)", r"\1\2", pred)
        ) or (
            re.sub(r"(^[A-ZÄÖÜ].+)e($)", r"\1\2", gold)
            == re.sub(r"(^[A-ZÄÖÜ].+)e($)", r"\1\2", pred)
        )
        other = not any([case_only, space_only, e_related])
        new_data["space_only"].append(space_only)
        new_data["case_only"].append(case_only)
        new_data["e-related"].append(e_related)
        new_data["other"].append(other)
    df = df.assign(**new_data)
    return df


def display_examples(
    df_sents: pd.DataFrame,
    df_errors: pd.DataFrame,
    error_id: int,
    max_num_examples: int = 5,
    max_tok_before: int = 3,
    max_tok_after: int = 3,
    seed: int = 42,
) -> List[Tuple[List[str], List[str], List[str]]]:
    # Look up error_id in df_errors and get orig and sent_id
    error_as_series = df_errors.loc[error_id]
    orig_type = error_as_series["orig"]
    sent_ids = list(error_as_series["sent_ids"])

    # Shuffle sent_ids and get first num_examples
    sent_ids.sort()
    random.Random(seed).shuffle(sent_ids)
    sent_ids = sent_ids[:max_num_examples]

    # Look up each sent_id in df_sents
    out = []
    for sent_id in sent_ids:
        selection = df_sents.loc[sent_id]
        # Find orig in sentence
        orig_tok = [orig for orig, _, _ in selection["alignm_orig2norm"]]
        gold_tok = [gold for _, gold, _ in selection["alignm_orig2norm"]]
        pred_tok = [pred for _, pred, _ in selection["alignm_orig2pred"]]
        # Only first occurrence is looked at
        token_index = orig_tok.index(orig_type)
        # Get selected spans (tokens before, tokens after) of the sentences
        start = max(token_index - max_tok_before, 0)
        end = token_index + max_tok_after + 1
        spans = (orig_tok[start:end], gold_tok[start:end], pred_tok[start:end])
        out.append(spans)

    for example in out:
        for version in example:
            print(version)
        print()

    return out


def errcat_aggregate_df(df_error_type: pd.DataFrame):
    """Returns a dataframe that contains the number of total errors per oov/invoc and the percentage of different error categories in either part"""

    multiindex = pd.MultiIndex.from_product(
        [["invoc", "oov"], ["total", "spacing", "casing", "-e-", "other"]]
    )

    total_invoc = df_error_type[df_error_type["invoc"] == True][  # noqa: E712
        "count"
    ].sum()
    total_invoc_space = df_error_type[
        (df_error_type["invoc"] == True)  # noqa: E712
        & (df_error_type["space_only"] == True)  # noqa: E712
    ]["count"].sum()
    total_invoc_case = df_error_type[
        (df_error_type["invoc"] == True)  # noqa: E712
        & (df_error_type["case_only"] == True)  # noqa: E712
    ]["count"].sum()
    total_invoc_e = df_error_type[
        (df_error_type["invoc"] == True)  # noqa: E712
        & (df_error_type["e-related"] == True)  # noqa: E712
    ]["count"].sum()
    total_invoc_other = df_error_type[
        (df_error_type["invoc"] == True)  # noqa: E712
        & (df_error_type["other"] == True)  # noqa: E712
    ]["count"].sum()
    total_oov = df_error_type[df_error_type["invoc"] == False][  # noqa: E712
        "count"
    ].sum()
    total_oov_space = df_error_type[
        (df_error_type["invoc"] == False)  # noqa: E712
        & (df_error_type["space_only"] == True)  # noqa: E712
    ]["count"].sum()
    total_oov_case = df_error_type[
        (df_error_type["invoc"] == False)  # noqa: E712
        & (df_error_type["case_only"] == True)  # noqa: E712
    ]["count"].sum()
    total_oov_e = df_error_type[
        (df_error_type["invoc"] == False)  # noqa: E712
        & (df_error_type["e-related"] == True)  # noqa: E712
    ]["count"].sum()
    total_oov_other = df_error_type[
        (df_error_type["invoc"] == False)  # noqa: E712
        & (df_error_type["other"] == True)  # noqa: E712
    ]["count"].sum()

    records = {
        multiindex[0]: total_invoc,
        multiindex[1]: total_invoc_space / total_invoc * 100,
        multiindex[2]: total_invoc_case / total_invoc * 100,
        multiindex[3]: total_invoc_e / total_invoc * 100,
        multiindex[4]: total_invoc_other / total_invoc * 100,
        multiindex[5]: total_oov,
        multiindex[6]: total_oov_space / total_oov * 100,
        multiindex[7]: total_oov_case / total_oov * 100,
        multiindex[8]: total_oov_e / total_oov * 100,
        multiindex[9]: total_oov_other / total_oov * 100,
    }

    df = pd.DataFrame.from_records([records], columns=multiindex)
    return df


def get_spans_of_unknown_tokens(
    text: str, tokenizer: PreTrainedTokenizerBase
) -> List[str]:
    """
    For a given string and tokenizer return the tokens that are unknown to the
    tokenizer

    This only works with huggingface "fast" tokenizers:
    https://huggingface.co/docs/tokenizers
    https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerFast
    """

    encoding = tokenizer(text)
    spans_unk_tokens = [
        # map a token index to a pair of character indices
        encoding.token_to_chars(token_index)[:]
        # we are only interested in the token indices of unknown tokens
        for token_index, token_id in enumerate(encoding["input_ids"])
        if token_id == tokenizer.unk_token_id
    ]
    return spans_unk_tokens
