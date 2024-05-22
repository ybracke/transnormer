#!/usr/bin/python
import Levenshtein
from align_levenshtein import align

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# YB's distillation of levenshtein_score without superfluous
# case distinction for align_type AND without caching (not supported for now)
def lev_norm_corpuslevel(sents_ref: List[str], sents_pred: List[str]) -> float:
    score = 0
    num_chars = 0
    for sent_ref, sent_pred in zip(sents_ref, sents_pred):
        sent_ref = sent_ref.replace("  ", " ")
        sent_pred = sent_pred.replace("  ", " ")
        score += Levenshtein.distance(sent_ref, sent_pred)
        num_chars += max(len(sent_pred), len(sent_ref))
    return score / num_chars


def word_acc_final(
    ref: List[str],
    pred: List[str],
    align_types: List[str] = ["both"],
    cache_file=None,
) -> Dict[str, Any]:
    """Computes accuracy metrics given a list of predictions and a list of references.

    Sentences with the same index get aligned on token-level for this.
    The `align_types` specify whether `ref` or `pred` is the base for the alignment.
    E.g., specifying `ref` aligns each token in `ref` to 0 or more tokens in `pred`.
    Include `"both"` in `align_types` to get the harmonized accuracy described in Bawden et al. (2022).

    The accuracy is computed over the entire corpus and per_sentence.

    Returns a dictionary like the following:
    {
        "ref": 0.5,
        "pred": 0.75,
        "both": 0.625,
        "per_sent": {
            "ref": [0.5, 0.5],
            "pred": [1.0, 0.5],
            "both": [0.75, 0.5],
        },
    }
    """

    scores: Dict[str, Optional[float]] = {
        "ref": None,
        "pred": None,
        "both": None,
    }
    per_sent_scores: Dict[str, np.ndarray] = {}

    # do this unless only 'pred' is chosen
    if align_types != ["pred"]:
        alignment_fwd = align(ref, pred, cache_file=cache_file)
        scores["ref"], per_sent_scores["ref"] = word_acc(alignment_fwd)

    # do this unless only 'ref' is chosen
    if align_types != ["ref"]:
        alignment_bckwd = align(pred, ref, cache_file=cache_file)
        scores["pred"], per_sent_scores["pred"] = word_acc(alignment_bckwd)

    if "both" in align_types:
        assert scores["ref"] is not None and scores["pred"] is not None
        scores["both"] = (scores["ref"] + scores["pred"]) / 2
        per_sent_scores["both"] = (per_sent_scores["ref"] + per_sent_scores["pred"]) / 2

    return {**scores, "per_sent": {k: list(v) for k, v in per_sent_scores.items()}}


def word_acc(
    alignments: List[List[Tuple[str, str, float]]]
) -> Tuple[float, np.ndarray]:
    """Accuracy (1) over the entire corpus and (2) per sentence"""
    scores = []
    correct_corpus, total_corpus = 0, 0
    for sent in alignments:
        correct_sent, total_sent = 0, 0
        for word in sent:
            # skip spaces
            if word[0] == "":
                continue

            if word[0] == word[1]:
                correct_sent += 1
            total_sent += 1
        scores.append(correct_sent / total_sent)
        correct_corpus += correct_sent
        total_corpus += total_sent
    if total_corpus == 0:
        return 0, np.array(scores)
    return correct_corpus / total_corpus, np.array(scores)
