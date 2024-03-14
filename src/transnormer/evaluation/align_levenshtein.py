#!/usr/bin/python
import pickle
import os
import re

from typing import List, Optional, Tuple

from .tokenise import basic_tokenise
from .wedit_distance_align import wedit_distance_align


def read_file(filename: str) -> List[str]:
    contents = []
    with open(filename) as fp:
        for line in fp:
            line = re.sub("[  ]", "  ", line.strip())
            #            contents.append(re.sub('[  ]+', ' ', basic_tokenise(line).strip()))
            contents.append(basic_tokenise(line).strip())
    return contents


# Helper function for the alignment, the actual strings are not altered
def homogenise(sent: str) -> str:
    sent = sent.lower()
    replace_from = "ǽǣáàâąãăåćčçďéèêëęěğìíîĩĭıïĺľłńñňòóôõøŕřśšşſťţùúûũǔỳýŷÿźẑżž"
    replace_into = "ææaaaaaaacccdeeeeeegiiiiiiilllnnnooooorrssssttuuuuuyyyyzzzz"
    table = sent.maketrans(replace_from, replace_into)
    return sent.translate(table)


def align(
    sents_ref: List[str], sents_pred: List[str], cache_file: Optional[str] = None
) -> List[List[Tuple[str, str, float]]]:
    """
    Align sentences in `sents_ref` and `sents_pred` on token-level.

    Sentences at the same index get aligned. Each token in a sentence from `sents_ref` gets aligned to 0 or more tokens in the corresponding sentence from `sents_pred`

    Tokens are understood as a stretch of characters without a space in-between. Thus you might want to apply some pre-processing to your text (e.g. with `basic_tokenise`).

    Example:

    ```python
    >>>align(['Sie bekommen ferner'], ['bekommen ferner an'])
    >>>[
        [
            ("Sie", "░", 4),
            ("bekommen", "bekommen", 0),
            ("ferner", "ferner▁an", 3.5999999999999996),
        ],
    ]
    """
    alignments, cache = [], {}
    if cache_file is not None and os.path.exists(cache_file):
        cache = pickle.load(open(cache_file, "rb"))
    for sent_ref, sent_pred in zip(sents_ref, sents_pred):
        if (sent_ref, sent_pred) in cache and "align" in cache[(sent_ref, sent_pred)]:
            alignment = cache[(sent_ref, sent_pred)]["align"]
            alignments.append(alignment)
        else:
            backpointers = wedit_distance_align(
                homogenise(sent_ref), homogenise(sent_pred)
            )

            alignment, current_word, seen1, seen2 = [], ["", ""], [], []
            last_weight: float = 0
            for i_ref, i_pred, weight in backpointers:
                if i_ref == 0 and i_pred == 0:
                    continue
                # spaces in both, add straight away
                if (
                    i_ref <= len(sent_ref)
                    and sent_ref[i_ref - 1] == " "
                    and i_pred <= len(sent_pred)
                    and sent_pred[i_pred - 1] == " "
                ):
                    alignment.append(
                        (
                            current_word[0].strip(),
                            current_word[1].strip(),
                            weight - last_weight,
                        )
                    )
                    last_weight = weight
                    current_word = ["", ""]
                    seen1.append(i_ref)
                    seen2.append(i_pred)
                else:
                    end_space = "░"
                    if i_ref <= len(sent_ref) and i_ref not in seen1:
                        if i_ref > 0:
                            current_word[0] += sent_ref[i_ref - 1]
                            seen1.append(i_ref)
                    if i_pred <= len(sent_pred) and i_pred not in seen2:
                        if i_pred > 0:
                            current_word[1] += (
                                sent_pred[i_pred - 1]
                                if sent_pred[i_pred - 1] != " "
                                else "▁"
                            )
                            end_space = "" if space_after(i_pred, sent_pred) else "░"
                            seen2.append(i_pred)
                    if (
                        i_ref <= len(sent_ref)
                        and sent_ref[i_ref - 1] == " "
                        and current_word[0].strip() != ""
                    ):
                        alignment.append(
                            (
                                current_word[0].strip(),
                                current_word[1].strip() + end_space,
                                weight - last_weight,
                            )
                        )
                        last_weight = weight
                        current_word = ["", ""]
            # final word
            alignment.append(
                (current_word[0].strip(), current_word[1].strip(), weight - last_weight)
            )
            # check that both strings are entirely covered
            recovered1 = re.sub(" +", " ", " ".join([x[0] for x in alignment]))
            recovered2 = re.sub(" +", " ", " ".join([x[1] for x in alignment]))

            assert recovered1 == re.sub(" +", " ", sent_ref), (
                "\n"
                + re.sub(" +", " ", recovered1)
                + "\n"
                + re.sub(" +", " ", sent_ref)
            )
            assert re.sub("[░▁ ]+", "", recovered2) == re.sub("[▁ ]+", "", sent_pred), (
                recovered2 + " / " + sent_pred
            )
            alignments.append(alignment)
            if cache is not None:
                if (sent_ref, sent_pred) not in cache:
                    cache[(sent_ref, sent_pred)] = {}
                cache[(sent_ref, sent_pred)]["align"] = alignment
    # dump cache if specified
    if cache_file is not None:
        pickle.dump(cache, open(cache_file, "wb"))
    return alignments


def space_after(idx: int, sent: str) -> bool:
    if idx < len(sent) - 1 and sent[idx + 1] == " ":
        return True
    return False


def space_before(idx: int, sent: str) -> bool:
    if idx > 0 and sent[idx - 1] == " ":
        return True
    return False


def prepare_for_print(alignments, print_weights=False) -> str:
    sents = []
    for align_sent in alignments:
        sent = ""
        for word1, word2, weight in align_sent:
            if word1 == word2:
                sent += word1 + " "
            else:
                if print_weights:
                    sent += word1 + "<|" + "{:.1f}".format(weight) + "|>" + word2 + " "
                else:
                    sent += word1 + "||||" + word2 + " "
        sents.append(sent.strip(" "))
    return "\n".join(sents)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("ref")
    parser.add_argument("pred")
    parser.add_argument(
        "-a",
        "--align_type",
        choices=("ref", "pred"),
        help="Which file's tokenisation to use as reference for alignment.",
    )
    parser.add_argument(
        "-c", "--cache", help="pickle cache file containing alignments", default=None
    )
    parser.add_argument(
        "-w",
        "--weights",
        help="replace |||| with <|weight|> in output",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    sents_ref, sents_pred = read_file(args.ref), read_file(args.pred)
    alignment = align(sents_ref, sents_pred, args.cache)
    print(prepare_for_print(alignment, args.weights))
