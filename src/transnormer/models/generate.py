import argparse
import os

from functools import partial
from typing import Dict, List, Optional, Union

import tomli
import random
import numpy as np
import torch
import datasets
import transformers

from transnormer.preprocess import translit
from transnormer.data.process import sort_dataset_by_length, filter_dataset_by_length


def set_seeds(n: int = 42):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)


# Generation function
def generate_normalization(
    batch: Dict,
    tokenizer: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
    device: torch.device,
    generation_config: transformers.GenerationConfig,
    tokenizer_kwargs: Dict[str, Union[bool, str, int]],
):
    input_strings = batch["orig"]
    inputs = tokenizer(
        input_strings,
        **tokenizer_kwargs,
        return_tensors="pt",
    ).to(device)

    outputs = model.generate(**inputs, generation_config=generation_config)
    output_strings = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_strings

    return batch


def load_data(
    path, n_examples: Optional[int] = None, split: Optional[str] = None
) -> datasets.Dataset:
    """
    Load the dataset.

    Dataset can be either a JSON file, a directory of JSON files or
    huggingface name of a dataset.
    """

    # Which split to use?
    s = split if split is not None else "train"
    if os.path.isfile(path):
        ds = datasets.load_dataset("json", data_files=path, split=s)
    elif os.path.isdir(path):
        ds = datasets.load_dataset("json", data_dir=path, split=s)
    else:
        try:
            ds = datasets.load_dataset(path, split=s)
        except datasets.exceptions.DatasetNotFoundError as e:
            raise e(f"Path '{path}' is no existing file or directory.")

    return ds


def parse_and_check_arguments(
    arguments: Optional[List[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generates normalizations given a configuration file that specifies the model, the data and parameters."
    )

    parser.add_argument(
        "-c",
        "--config",
        default="test_config.toml",
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

    # (1) Load configs
    with open(args.config, mode="rb") as fp:
        CONFIGS = tomli.load(fp)

    # (2) Preparations
    # (2.1) Set fixed seed (random, numpy, torch) for reproducibilty
    set_seeds(CONFIGS["random_seed"])

    # (2.2) GPU set-up
    gpu_index = CONFIGS.get("gpu")
    device = torch.device(
        gpu_index if gpu_index is not None and torch.cuda.is_available() else "cpu"
    )

    # (3) Data
    data_path = CONFIGS["data"]["path_test"]
    split = CONFIGS["data"].get("split")
    ds = load_data(data_path, split)

    # (4) Tokenizers and transliterator
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        CONFIGS["tokenizer"]["checkpoint_in"]
    )

    # Optional: replace tokenizer's normalization component with a custom transliterator
    if "input_transliterator" in CONFIGS["tokenizer"]:
        if CONFIGS["tokenizer"]["input_transliterator"] == "Transliterator1":
            transliterator = translit.Transliterator1()
        else:
            transliterator = None
        if transliterator:
            tokenizer = translit.exchange_transliterator(tokenizer, transliterator)

    # (5) Load model
    checkpoint = CONFIGS["model"]["checkpoint"]
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)

    # (6) Data preparation
    # Sort by length
    index_column = "#"
    len_column = "len"
    ds = sort_dataset_by_length(
        ds,
        "orig",
        descending=True,
        name_index_column=index_column,
        name_length_column=len_column,
        use_bytelength=True,
    )

    # Optional: Filter out samples that exceed given length
    k = CONFIGS["data"].get("max_bytelength")
    if k:
        ds = filter_dataset_by_length(ds, max_length=k, name_length_column=len_column)

    # Optional: Clip dataset to fixed number of samples
    n = CONFIGS["data"].get("n_examples_test")
    if n:
        ds = ds.shuffle().select(range(n))

    # (7) Generation
    # Parameters for model output
    gen_cfg = transformers.GenerationConfig(**CONFIGS["generation_config"])

    # Prepare generation function as a partial function (only batch missing)
    normalize = partial(
        generate_normalization,
        tokenizer=tokenizer,
        model=model,
        device=device,
        generation_config=gen_cfg,
        tokenizer_kwargs=CONFIGS["tokenizer_configs"],
    )

    # Call generation function
    ds = ds.map(
        normalize,
        batched=True,
        batch_size=CONFIGS["generation"]["batch_size"],
        load_from_cache_file=False,
    )

    # Sort in original order
    ds = ds.sort(index_column)
    ds = ds.remove_columns([index_column, len_column])

    # (8) Save outputs
    ds.to_json(args.out, force_ascii=False)


if __name__ == "__main__":
    main()
