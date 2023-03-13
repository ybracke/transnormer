from datetime import datetime
import os
import random
import time
import tomli

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer

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

    splits_and_paths = [
        ("train", CONFIGS["data"]["paths_train"]),
        ("validation", CONFIGS["data"]["paths_validation"]),
        ("test", CONFIGS["data"]["paths_test"]),
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
        num_examples = CONFIGS["data"][f"n_examples_{split}"]
        ds2num_examples = {dsets: num_examples[i] for i, dsets in enumerate(dsets)}
        # Merge and resample datasets for this split
        ds = loader.merge_datasets(ds2num_examples, seed=CONFIGS["random_seed"])
        ds_split_merged[split] = ds

    dataset = ds_split_merged

    # Create smaller datasets from random examples
    if "subset_sizes" in CONFIGS:
        train_size = CONFIGS["subset_sizes"]["train"]
        validation_size = CONFIGS["subset_sizes"]["validation"]
        test_size = CONFIGS["subset_sizes"]["test"]
        dataset["train"] = dataset["train"].shuffle().select(range(train_size))
        dataset["validation"] = (
            dataset["validation"].shuffle().select(range(validation_size))
        )
        dataset["test"] = dataset["test"].shuffle().select(range(test_size))

    # (3) Tokenize data

    print("Tokenizing and preparing the data ...")
    start = time.process_time()

    # Load tokenizers
    tokenizer_input = AutoTokenizer.from_pretrained(
        CONFIGS["language_models"]["checkpoint_encoder"]
    )
    # Replace tokenizer's normalization component with a custom transliterator
    if "input_transliterator" in CONFIGS["tokenizer"]:
        from transnormer.preprocess import translit

        if CONFIGS["tokenizer"]["input_transliterator"] == "Transliterator1":
            transliterator = translit.Transliterator1()
        else:
            transliterator = None
        tokenizer_input = translit.exchange_transliterator(
            tokenizer_input, transliterator
        )

    tokenizer_output = AutoTokenizer.from_pretrained(
        CONFIGS["language_models"]["checkpoint_decoder"]
    )

    tokenization_kwargs = {
        # FIXME: it is unclear how/if a custom tokenizer can be passed as a parameter
        "tokenizer_input": tokenizer_input,
        "tokenizer_output": tokenizer_output,
        "max_length_input": CONFIGS["tokenizer"]["max_length_input"],
        "max_length_output": CONFIGS["tokenizer"]["max_length_output"],
    }

    # Tokenize by applying a mapping
    prepared_dataset = dataset.map(
        tokenize_input_and_output,
        fn_kwargs=tokenization_kwargs,
        remove_columns=["orig", "norm"],
        batched=True,
        batch_size=CONFIGS["training_hyperparams"]["batch_size"],
    )

    # Convert to torch tensors
    prepared_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
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
