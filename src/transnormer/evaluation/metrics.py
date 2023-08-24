#!/usr/bin/python
import Levenshtein
from transnormer.evaluation.align_levenshtein import align

from typing import Any, Dict, List, Optional, Tuple


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
    ref: List[str], pred: List[str], align_types: List[str], cache_file=None
) -> Dict[str, Any]:
    scores: Dict[str, Optional[float]] = {"ref": None, "pred": None, "both": None}

    # do this unless only 'pred' is chosen
    if align_types != ["pred"]:
        alignment_fwd = align(ref, pred, cache_file=cache_file)
        scores["ref"] = word_acc(alignment_fwd)

    # do this unless only 'ref' is chosen
    if align_types != ["ref"]:
        alignment_bckwd = align(pred, ref, cache_file=cache_file)
        scores["pred"] = word_acc(alignment_bckwd)

    if "both" in align_types:
        assert scores["ref"] is not None and scores["pred"] is not None
        scores["both"] = (scores["ref"] + scores["pred"]) / 2

    return scores


def word_acc(alignments: List[List[Tuple[str, str, float]]]) -> float:
    correct, total = 0, 0
    for sent in alignments:
        for word in sent:
            # skip spaces
            if word[0] == "":
                continue

            if word[0] == word[1]:
                correct += 1
            total += 1
    return correct / total
