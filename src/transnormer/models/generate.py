import argparse
from typing import List, Optional

import tomli
import random
import numpy as np
import torch
import datasets
import transformers

from transnormer.preprocess import translit
from transnormer.data.process import sort_dataset_by_length


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

    # (1) Load configs
    with open(args.config, mode="rb") as fp:
        CONFIGS = tomli.load(fp)

    # (2) Preparations
    # (2.1) Fix seeds for reproducibilty
    random.seed(CONFIGS["random_seed"])
    np.random.seed(CONFIGS["random_seed"])
    torch.manual_seed(CONFIGS["random_seed"])

    # (2.2) GPU set-up
    device = torch.device(CONFIGS["gpu"] if torch.cuda.is_available() else "cpu")

    # (3) Data
    path = CONFIGS["data"]["path_test"]
    ds = datasets.load_dataset("json", data_files=path)

    # Optional: Take only N examples
    if "n_examples_test" in CONFIGS["data"]:
        n = CONFIGS["data"]["n_examples_test"]
        # Note: "train" is the default dataset name assigned
        ds["train"] = ds["train"].shuffle().select(range(n))

    # FIXME: only for 0_TEST Remove and rename columns
    # ds["train"] = ds["train"].remove_columns(["author", "basename", "title", "date", "genre", "norm", "par_id", "done"])
    # ds["train"] = ds["train"].rename_column("text", "orig")
    # ds["train"] = ds["train"].rename_column("norm_manual", "norm")

    # (4) Tokenizers and transliterator
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

    # (5) Load model
    checkpoint = CONFIGS["model"]["checkpoint"]
    config = transformers.AutoConfig.from_pretrained(checkpoint)
    # HOTFIX for using byt5
    if config.architectures.pop() == "T5ForConditionalGeneration":
        model = transformers.T5ForConditionalGeneration.from_pretrained(checkpoint).to(
            device
        )
    else:
        model = transformers.EncoderDecoderModel.from_pretrained(checkpoint).to(device)

    # (6) Generation
    # Parameters for model output
    gen_cfg = transformers.GenerationConfig(**CONFIGS["generation_config"])

    # Generation function
    def generate_normalization(batch):
        inputs = tokenizer_input(
            batch["orig"],
            **CONFIGS["tokenizer_configs"],
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        outputs = model.generate(
            input_ids, attention_mask=attention_mask, generation_config=gen_cfg
        )
        output_str = tokenizer_output.batch_decode(outputs, skip_special_tokens=True)

        batch["pred"] = output_str

        return batch

    # Sort by length
    dataset = ds["train"]
    dataset = sort_dataset_by_length(
        dataset, "orig", descending=True, keep_length_column=False
    )
    ds["train"] = dataset

    # Call generation function
    ds = ds.map(
        generate_normalization,
        batched=True,
        batch_size=CONFIGS["generation"]["batch_size"],
        load_from_cache_file=False,
    )

    # (7) Save outputs
    ds["train"].to_json(args.out, force_ascii=False)


if __name__ == "__main__":
    main()
