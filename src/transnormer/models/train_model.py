from datetime import datetime
import os
import random
import time
import tomli

from typing import Dict, Any, Tuple

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
import transformers
from transnormer.data import loader
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

    # (1.1) Load input tokenizer
    tokenizer_input = transformers.AutoTokenizer.from_pretrained(
        configs["language_models"]["checkpoint_encoder"]
    )
    # (1.2) Optional: replace tokenizer's normalization component with a custom transliterator
    if "input_transliterator" in configs["tokenizer"]:
        if configs["tokenizer"]["input_transliterator"] == "Transliterator1":
            transliterator = translit.Transliterator1()
        else:
            transliterator = None
        tokenizer_input = translit.exchange_transliterator(
            tokenizer_input, transliterator
        )
    # (2) Load output tokenizer
    tokenizer_output = transformers.AutoTokenizer.from_pretrained(
        configs["language_models"]["checkpoint_decoder"]
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
        batch_size=configs["training_hyperparams"]["batch_size"],
    )

    # Convert to torch tensors
    prepared_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    return prepared_dataset, tokenizer_input, tokenizer_output


def load_and_merge_datasets(configs: Dict[str, Any]) -> datasets.DatasetDict:
    """
    Load, resample and merge the dataset splits as specified in the config file.
    """

    splits_and_paths = [
        ("train", configs["data"]["paths_train"]),
        ("validation", configs["data"]["paths_validation"]),
        ("test", configs["data"]["paths_test"]),
    ]

    ds_split_merged = datasets.DatasetDict()
    # Iterate over splits (i.e. train, validation, test)
    for split, paths in splits_and_paths:
        # Load all datasets for this split
        dsets = [
            datasets.load_dataset("json", data_files=path, split="train")
            for path in paths
        ]
        # Map each dataset to the desired number of examples for this dataset
        num_examples = configs["data"][f"n_examples_{split}"]
        ds2num_examples = {dsets: num_examples[i] for i, dsets in enumerate(dsets)}
        # Merge and resample datasets for this split
        ds = loader.merge_datasets(ds2num_examples, seed=configs["random_seed"])
        ds_split_merged[split] = ds

    dataset = ds_split_merged

    # Optional: create smaller datasets from random examples
    if "subset_sizes" in configs:
        train_size = configs["subset_sizes"]["train"]
        validation_size = configs["subset_sizes"]["validation"]
        test_size = configs["subset_sizes"]["test"]
        dataset["train"] = dataset["train"].shuffle().select(range(train_size))
        dataset["validation"] = (
            dataset["validation"].shuffle().select(range(validation_size))
        )
        dataset["test"] = dataset["test"].shuffle().select(range(test_size))

    return dataset


# not used yet
def train(
    model: transformers.BertForMaskedLM,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Training loop"""

    return None


# TODO pass this on the command-line
ROOT = "/home/bracke/code/transnormer"
CONFIGFILE = os.path.join(ROOT, "training_config.toml")


if __name__ == "__main__":
    # (1) Preparations
    # Load configs
    with open(CONFIGFILE, mode="rb") as fp:
        CONFIGS = tomli.load(fp)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    MODELDIR = os.path.join(ROOT, f"./models/models_{timestamp}")

    # Fix seeds for reproducibilty
    random.seed(CONFIGS["random_seed"])
    np.random.seed(CONFIGS["random_seed"])
    torch.manual_seed(CONFIGS["random_seed"])

    # GPU set-up
    device = torch.device(CONFIGS["gpu"] if torch.cuda.is_available() else "cpu")

    # (2) Load data

    print("Loading the data ...")
    dataset = load_and_merge_datasets(CONFIGS)

    # (3) Tokenize data

    print("Tokenizing and preparing the data ...")
    start = time.process_time()
    prepared_dataset, tokenizer_input, tokenizer_output = tokenize_datasets(
        dataset, CONFIGS
    )
    end = time.process_time()
    print(f"Elapsed time for tokenization: {end - start}")

    # (4) Load models

    print("Loading the pre-trained models ...")

    model = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(
        CONFIGS["language_models"]["checkpoint_encoder"],
        CONFIGS["language_models"]["checkpoint_decoder"],
    ).to(device)

    # Setting the special tokens
    model.config.decoder_start_token_id = tokenizer_output.cls_token_id
    model.config.eos_token_id = tokenizer_output.sep_token_id
    model.config.pad_token_id = tokenizer_output.pad_token_id
    # model.config.vocab_size = model.config.encoder.vocab_size # TODO

    # Params for beam search decoding
    model.config.max_length = CONFIGS["tokenizer"]["max_length_output"]
    model.config.no_repeat_ngram_size = CONFIGS["beam_search_decoding"][
        "no_repeat_ngram_size"
    ]
    model.config.early_stopping = CONFIGS["beam_search_decoding"]["early_stopping"]
    model.config.length_penalty = CONFIGS["beam_search_decoding"]["length_penalty"]
    model.config.num_beams = CONFIGS["beam_search_decoding"]["num_beams"]

    # (5) Training

    print("Training ...")

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=MODELDIR,
        predict_with_generate=True,
        evaluation_strategy=CONFIGS["training_hyperparams"]["eval_strategy"],
        fp16=CONFIGS["training_hyperparams"]["fp16"],
        eval_steps=CONFIGS["training_hyperparams"]["eval_steps"],
        num_train_epochs=CONFIGS["training_hyperparams"]["epochs"],
        per_device_train_batch_size=CONFIGS["training_hyperparams"]["batch_size"],
        per_device_eval_batch_size=CONFIGS["training_hyperparams"]["batch_size"],
        save_steps=CONFIGS["training_hyperparams"][
            "save_steps"
        ],  # Does this work? (custom tokenizer can't be saved)
    )

    # Instantiate trainer
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset["train"],
        eval_dataset=prepared_dataset["validation"],
        # compute_metrics=compute_metrics, # TODO
    )

    trainer.train()

    # (6) Saving the final model

    model_path = os.path.join(MODELDIR, "model_final/")
    model.save_pretrained(model_path)
    # this fails because a custom tokenizer can't be saved
    # model_path = f"./models/model_fromtrainer/"
    # trainer.save_model(model_path)
