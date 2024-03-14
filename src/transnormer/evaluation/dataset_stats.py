import pickle
import random
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import tomli
import torch

from transnormer.evaluation import align_levenshtein, tokenise
from transnormer.models import train_model


def type_alignment_stats(
    alignment: List[List[Tuple[str, str, float]]]
) -> Dict[str, Dict[str, int]]:
    """
    Get the statistics for the token alignments.

    Alignments have to be computed like this:
    ```
    align_levenshtein.align(
        [tokenise.basic_tokenise(sent) for sent in sents_src],
        [tokenise.basic_tokenise(sent) for sent in sents_trg],
    )
    ```
    """

    stats = {}
    for sent in alignment:
        for src, trg, _ in sent:
            # Source type not in dict yet: create an inner dict with first target
            if src not in stats:
                stats[src] = {trg: 1}
            # Otherwise: add target type in inner dict
            else:
                trg_cnts = stats[src]
                if trg in trg_cnts:
                    trg_cnts[trg] += 1
                else:
                    trg_cnts[trg] = 1

    return stats


def get_typestats_for_training_data(
    configfile: str, outfile: str, src: str = "orig"
) -> None:
    """
    Pass a path to a model and get the stats for the training data that belongs to this model.

    The training data processing will be recreated from the training_config.toml.

    `src` must be one of {"orig", "norm"}
    """

    # Get the training data from training_config.toml
    print("Loading and processing the training data ...")
    # (1) Preparations
    # Load configs
    with open(configfile, mode="rb") as fp:
        CONFIGS = tomli.load(fp)

    # Fix seeds for reproducibilty
    random.seed(CONFIGS["random_seed"])
    np.random.seed(CONFIGS["random_seed"])
    torch.manual_seed(CONFIGS["random_seed"])

    # GPU set-up
    device = torch.device(CONFIGS["gpu"] if torch.cuda.is_available() else "cpu")
    # limit memory usage to 80%
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8, device)

    # (2) Load data
    dataset_dict = train_model.load_and_merge_datasets(CONFIGS)

    # (3) Tokenize data
    tokenizer_input, tokenizer_output = train_model.load_tokenizers(CONFIGS)
    prepared_dataset_dict = train_model.tokenize_dataset_dict(
        dataset_dict, tokenizer_input, tokenizer_output, CONFIGS
    )

    # (3.1) Optional: Filter data for length
    prepared_dataset_dict = train_model.filter_dataset_dict_for_length(
        prepared_dataset_dict, CONFIGS
    )

    if src == "orig":
        src_sents = prepared_dataset_dict["train"]["orig"]
        trg_sents = prepared_dataset_dict["train"]["norm"]
    elif src == "norm":
        src_sents = prepared_dataset_dict["train"]["norm"]
        trg_sents = prepared_dataset_dict["train"]["orig"]
    else:
        raise ValueError("Argument `src` must be one of {'norm', 'orig'}.")

    print(f"Computing the alignments. Current time: {datetime.now().time()}")
    alignment = align_levenshtein.align(
        [tokenise.basic_tokenise(sent) for sent in src_sents],
        [tokenise.basic_tokenise(sent) for sent in trg_sents],
    )
    print(f"Done computing the alignments. Current time: {datetime.now().time()}")
    stats = type_alignment_stats(alignment)

    with open(outfile, "wb") as f:
        pickle.dump(stats, f)

    return
