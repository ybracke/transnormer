from datetime import datetime
import math
import os
import random
import shutil
from typing import Any, Dict


import tomli
import datasets
import numpy as np
import torch

import transformers
from transnormer.data import loader, process


def tokenize_input_and_output(
    batch: Dict,
    tokenizer: transformers.PreTrainedTokenizerBase,
    reverse_labels: bool = False,
) -> Dict:
    """
    Tokenizes a `batch` of input and label strings. Assumes that input string
    (label string) is stored in batch under the key `"orig"` (`"norm"`).

    If reverse_labels=True, `"orig"` and `"norm"` are switched. Use this for training
    a model that produces reversed predictions, e.g. modern->historical

    Function is inspired by `process_data_to_model_inputs` described here:
    https://huggingface.co/blog/warm-starting-encoder-decoder#warm-starting-the-encoder-decoder-model
    """

    # Tokenize the inputs and labels
    inputs = (
        tokenizer(batch["orig"]) if not reverse_labels else tokenizer(batch["norm"])
    )
    outputs = (
        tokenizer(batch["norm"]) if not reverse_labels else tokenizer(batch["orig"])
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # Make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch


def load_tokenizer(configs: Dict[str, Any]) -> transformers.PreTrainedTokenizerBase:
    """Load tokenizer object based on config dictionary"""
    tokenizer = None

    # (A) If tokenizer is given explicitly in config file
    if "tokenizer" in configs["tokenizer"]:
        # Load input tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            configs["tokenizer"]["tokenizer"]
        )
        return tokenizer

    # (B) tokenizer not explicitly stated in config file, but via language model
    # (1) Load tokenizers
    if "checkpoint_encoder_decoder" in configs["language_models"]:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            configs["language_models"]["checkpoint_encoder_decoder"]
        )

    return tokenizer


def tokenize_dataset_dict(
    dataset_dict: datasets.DatasetDict,
    tokenizer: transformers.PreTrainedTokenizerBase,
    configs,
) -> datasets.DatasetDict:
    """
    Tokenize the datasets in a DatasetDict as specified in the config file.

    """

    # Tokenize by applying map function to the DatasetDict
    prepared_dataset_dict = dataset_dict.map(
        tokenize_input_and_output,
        fn_kwargs={
            "tokenizer": tokenizer,
            "reverse_labels": configs["data"].get("reverse_labels", False),
        },
        batched=True,
        batch_size=configs["training_hyperparams"]["batch_size"],
        load_from_cache_file=False,
    )

    # Convert to torch tensors
    prepared_dataset_dict.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    return prepared_dataset_dict


def filter_dataset_dict_for_length(
    dataset_dict: datasets.DatasetDict,
    configs: Dict,
) -> datasets.DatasetDict:
    """Add a length column based on "input_ids" and filter out examples that have
    lengths out of a given range"""
    min_length = configs["tokenizer"].get("min_length_input", 0)
    max_length = configs["tokenizer"].get("max_length_input", -1)

    for split, dataset in dataset_dict.items():
        lengths = [len(s) for s in dataset["input_ids"]]
        dataset = dataset.add_column("length", lengths)
        dataset = process.filter_dataset_by_length(dataset, max_length, min_length)
        dataset_dict[split] = dataset

    return dataset_dict


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
        ds = loader.merge_datasets(
            ds2num_examples,
            seed=configs["random_seed"],
            shuffle=configs["data"].get("do_shuffle", True),
        )
        ds_split_merged[split] = ds

    dataset = ds_split_merged

    return dataset


def warmstart_seq2seq_model(
    configs: Dict[str, Any],
    tokenizer: transformers.PreTrainedTokenizerBase,
    device: torch.device,
) -> transformers.PreTrainedModel:
    """
    Load and configure an encoder-decoder model.
    """

    model = transformers.T5ForConditionalGeneration.from_pretrained(
        configs["language_models"]["checkpoint_encoder_decoder"],
    ).to(device)

    # Setting the special tokens
    model.config.decoder_start_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model


def train_seq2seq_model(
    model: transformers.PreTrainedModel,
    train_dataset: datasets.Dataset,
    eval_dataset: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizerBase,
    configs: Dict[str, Any],
    output_dir: str,
) -> None:
    """
    Train an encoder-decoder model with given configurations.

    `model` will be altered by this function.
    """

    # Set-up training arguments from hyperparameters
    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=output_dir,
        predict_with_generate=True,
        num_train_epochs=configs["training_hyperparams"]["epochs"],
        per_device_train_batch_size=configs["training_hyperparams"]["batch_size"],
        per_device_eval_batch_size=configs["training_hyperparams"]["batch_size"],
        fp16=configs["training_hyperparams"]["fp16"],
        group_by_length=True,
        save_strategy=configs["training_hyperparams"]["save_strategy"],
        logging_strategy=configs["training_hyperparams"]["logging_strategy"],
        evaluation_strategy=configs["training_hyperparams"]["eval_strategy"],
        load_best_model_at_end=True,
    )

    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, padding=configs["tokenizer"]["padding"]
    )

    class CustomCallback(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            """Evaluate and log every half epoch"""
            # Determine steps per epoch
            steps_per_epoch = math.ceil(
                len(train_dataset) / (args.per_device_train_batch_size * args._n_gpu)
            )
            # Trigger evaluation only at the middle of the epoch
            # Combine this with {eval,logging}_strategy="epoch"
            if state.global_step % steps_per_epoch == steps_per_epoch // 2:
                control.should_log = True
                control.should_evaluate = True

    # Instantiate trainer
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[CustomCallback],
    )

    # Run training
    train_result = trainer.train()

    # After training
    trainer.save_model()
    trainer.create_model_card()
    trainer.save_state()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    metrics = trainer.evaluate(metric_key_prefix="eval")
    metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    return None


def main():
    ROOT = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../..")
    )
    CONFIGFILE = os.path.join(ROOT, "training_config.toml")

    # (1) Preparations
    # Load configs
    with open(CONFIGFILE, mode="rb") as fp:
        CONFIGS = tomli.load(fp)
    outdir = CONFIGS["training_hyperparams"].get("output_dir")
    if outdir is not None:
        if os.path.isabs(outdir):
            MODELDIR = outdir
        else:
            MODELDIR = os.path.join(ROOT, outdir)
    else:
        MODELDIR = os.path.join(
            ROOT, f"./models/models_{datetime.today().strftime('%Y-%m-%d')}"
        )
    if not os.path.isdir(MODELDIR):
        os.makedirs(MODELDIR)

    # Save the config file to model directory
    shutil.copy(CONFIGFILE, MODELDIR)

    # Fix seeds for reproducibilty
    random.seed(CONFIGS["random_seed"])
    np.random.seed(CONFIGS["random_seed"])
    torch.manual_seed(CONFIGS["random_seed"])

    # GPU set-up
    gpu_index = CONFIGS.get("gpu")
    device = torch.device(
        gpu_index if gpu_index is not None and torch.cuda.is_available() else "cpu"
    )
    # limit memory usage to 90%
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9, device)

    # (2) Load data

    print("Loading the data ...")
    dataset_dict = load_and_merge_datasets(CONFIGS)

    # (3) Tokenize data
    print("Tokenizing and preparing the data ...")
    tokenizer = load_tokenizer(CONFIGS)
    prepared_dataset_dict = tokenize_dataset_dict(dataset_dict, tokenizer, CONFIGS)

    # (3.1) Optional: Filter data for length
    prepared_dataset_dict = filter_dataset_dict_for_length(
        prepared_dataset_dict, CONFIGS
    )

    # (4) Load models
    print("Loading the pre-trained models ...")
    model = warmstart_seq2seq_model(CONFIGS, tokenizer, device)

    # (5) Training

    print("Training model ...")
    train_seq2seq_model(
        model,
        prepared_dataset_dict["train"],
        prepared_dataset_dict["validation"],
        tokenizer,
        CONFIGS,
        MODELDIR,
    )


if __name__ == "__main__":
    main()
