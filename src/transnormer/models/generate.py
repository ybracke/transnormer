import argparse
from typing import Any, Dict, List, Optional, Tuple

import tomli
import random
import numpy as np
import torch
import datasets
import pandas as pd
import transformers

from transnormer.preprocess import translit


def tokenize_input_and_output(
    batch, tokenizer_input, tokenizer_output, max_length_input, max_length_output
):
    """
    Tokenizes a `batch` of input and label strings. Assumes that input string
    (label string) is stored in batch under the key `"orig"` (`"norm"`).

    Function is inspired by `process_data_to_model_inputs` described here:
    https://huggingface.co/blog/warm-starting-encoder-decoder#warm-starting-the-encoder-decoder-model
    """

    # Tokenize the inputs and labels
    inputs = tokenizer_input(
        batch["orig"],
        padding="max_length",
        truncation=True,
        max_length=max_length_input,
    )
    outputs = tokenizer_output(
        batch["norm"],
        padding="max_length",
        truncation=True,
        max_length=max_length_output,
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # Make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer_output.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch


# FIXME: For simplicity I just copied and adjusted this function from train_model.py
# TODO: Adjust it so that it can be used in both training and prediction
def tokenize_datasets(
    dataset: datasets.DatasetDict, configs: Dict[str, Any]
) -> Tuple[
    datasets.DatasetDict,
    transformers.PreTrainedTokenizerBase,
    transformers.PreTrainedTokenizerBase,
]:
    """
    Tokenize the datasets in a DatasetDict as specified in the config file.

    Also returns the loaded input and output tokenizers.
    """

    # (1) Load tokenizers
    #  (1.1) Load input and output tokenizer
    tokenizer_input = transformers.AutoTokenizer.from_pretrained(
        configs["tokenizer"]["checkpoint_in"]
    )
    if "checkpoint_out" in configs["tokenizer"]:
        tokenizer_output = transformers.AutoTokenizer.from_pretrained(
            configs["tokenizer"]["checkpoint_out"]
        )
    else:
        # Output tokenizer is simply a reference to input tok
        tokenizer_output = tokenizer_input

    # (1.2) Optional: replace tokenizer's normalization component with a custom transliterator
    if "input_transliterator" in configs["tokenizer"]:
        if configs["tokenizer"]["input_transliterator"] == "Transliterator1":
            transliterator = translit.Transliterator1()
        else:
            transliterator = None
        tokenizer_input = translit.exchange_transliterator(
            tokenizer_input, transliterator
        )

    # (3) Define tokenization keyword arguments
    tokenization_kwargs = {
        "tokenizer_input": tokenizer_input,
        "tokenizer_output": tokenizer_output,
        "max_length_input": configs["tokenizer"]["max_length_input"],
        "max_length_output": configs["tokenizer"]["max_length_output"],
    }

    # Tokenize by applying map function to the DatasetDict
    prepared_dataset = dataset.map(
        tokenize_input_and_output,
        fn_kwargs=tokenization_kwargs,
        remove_columns=["orig", "norm"],
        batched=True,
        batch_size=configs["generation"]["batch_size"],
        load_from_cache_file=False,
    )

    # Convert to torch tensors
    prepared_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    return prepared_dataset, tokenizer_input, tokenizer_output


def parse_and_check_arguments(
    arguments: Optional[List[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generates normalizations given a configuration file that specifies the model, the data and parameters."
    )

    # TODO: allow to overwrite configs on command-line?
    parser.add_argument(
        "-c",
        "--config",
        help="Path to the config file (TOML)",
    )
    parser.add_argument(
        "-o",
        "--out",
        help="Path to the output file (JSONL)",
    )

    args = parser.parse_args(arguments)

    return args


def main(arguments: Optional[List[str]] = None) -> None:
    args = parse_and_check_arguments(arguments)

    # Load configs
    with open(args.config, mode="rb") as fp:
        CONFIGS = tomli.load(fp)

    # Fix seeds for reproducibilty
    random.seed(CONFIGS["random_seed"])
    np.random.seed(CONFIGS["random_seed"])
    torch.manual_seed(CONFIGS["random_seed"])

    # GPU set-up
    device = torch.device(CONFIGS["gpu"] if torch.cuda.is_available() else "cpu")

    path = CONFIGS["data"]["path_test"]
    ds = datasets.load_dataset("json", data_files=path)

    # Take only N examples
    n = CONFIGS["data"]["n_examples_test"]
    # NB: "train" is the default dataset name assigned
    ds["train"] = ds["train"].shuffle().select(range(n))

    # FIXME: Remove and rename columns (for 0_TEST)
    ds["train"] = ds["train"].remove_columns(
        ["author", "basename", "title", "date", "genre", "norm", "par_id", "done"]
    )
    ds["train"] = ds["train"].rename_column("text", "orig")
    ds["train"] = ds["train"].rename_column("norm_manual", "norm")

    # Tokenize data
    prepared_dataset, tokenizer_input, tokenizer_output = tokenize_datasets(ds, CONFIGS)

    # Load model
    checkpoint = CONFIGS["model"]["checkpoint"]
    model = transformers.EncoderDecoderModel.from_pretrained(checkpoint).to(device)

    # Parameters for model output
    model.config.max_length = CONFIGS["tokenizer"]["max_length_output"]
    model.config.early_stopping = CONFIGS["beam_search_decoding"]["early_stopping"]
    model.config.length_penalty = CONFIGS["beam_search_decoding"]["length_penalty"]
    model.config.num_beams = CONFIGS["beam_search_decoding"]["num_beams"]

    # Generate
    def generate_normalization(batch):
        inputs = tokenizer_input(
            batch["orig"],
            padding="max_length",
            truncation=True,
            max_length=CONFIGS["tokenizer"]["max_length_input"],
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        outputs = model.generate(input_ids, attention_mask=attention_mask)
        output_str = tokenizer_output.batch_decode(outputs, skip_special_tokens=True)

        batch["pred"] = output_str

        return batch

    ds = ds.map(
        generate_normalization,
        batched=True,
        batch_size=8,
        load_from_cache_file=False,
    )

    ds["train"].to_json(args.out, force_ascii=False)


if __name__ == "__main__":
    main()
