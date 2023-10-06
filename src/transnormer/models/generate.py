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

    # Optional: Take only N examples
    if "n_examples_test" in CONFIGS["data"]:
        n = CONFIGS["data"]["n_examples_test"]
        # NB: "train" is the default dataset name assigned
        ds["train"] = ds["train"].shuffle().select(range(n))

    # FIXME: only for 0_TEST Remove and rename columns
    # ds["train"] = ds["train"].remove_columns(
    # ["author", "basename", "title", "date", "genre", "norm", "par_id", "done"]
    # )
    # ds["train"] = ds["train"].rename_column("text", "orig")
    # ds["train"] = ds["train"].rename_column("norm_manual", "norm")

    # Load tokenizer(s)
    tokenizer_input = transformers.AutoTokenizer.from_pretrained(
        CONFIGS["tokenizer"]["checkpoint_in"]
    )
    if "checkpoint_out" in CONFIGS["tokenizer"]:
        tokenizer_output = transformers.AutoTokenizer.from_pretrained(
            CONFIGS["tokenizer"]["checkpoint_out"]
        )
    else:
        # Output tokenizer is simply a reference to input tok
        tokenizer_output = tokenizer_input

    # Optional: replace tokenizer's normalization component with a custom transliterator
    if "input_transliterator" in CONFIGS["tokenizer"]:
        if CONFIGS["tokenizer"]["input_transliterator"] == "Transliterator1":
            transliterator = translit.Transliterator1()
        else:
            transliterator = None
        tokenizer_input = translit.exchange_transliterator(
            tokenizer_input, transliterator
        )

    # Load model
    checkpoint = CONFIGS["model"]["checkpoint"]
    config = transformers.AutoConfig.from_pretrained(checkpoint)
    # HOTFIX for using byt5
    if config.architectures.pop() == "T5ForConditionalGeneration":
        model = transformers.T5ForConditionalGeneration.from_pretrained(checkpoint).to(
            device
        )
    else:
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
        batch_size=CONFIGS["generation"]["batch_size"],
        load_from_cache_file=False,
    )

    ds["train"].to_json(args.out, force_ascii=False)


if __name__ == "__main__":
    main()
