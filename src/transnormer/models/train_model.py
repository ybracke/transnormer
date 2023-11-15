import json
import os
import random
import shutil
from typing import Any, Dict, Tuple


import tomli
import datasets
import numpy as np
import torch

import transformers
from transnormer.data import loader, process
from transnormer.preprocess import translit


def tokenize_input_and_output(
    batch: Dict,
    tokenizer_input: transformers.PreTrainedTokenizerBase,
    tokenizer_output: transformers.PreTrainedTokenizerBase,
) -> Dict:
    """
    Tokenizes a `batch` of input and label strings. Assumes that input string
    (label string) is stored in batch under the key `"orig"` (`"norm"`).

    Function is inspired by `process_data_to_model_inputs` described here:
    https://huggingface.co/blog/warm-starting-encoder-decoder#warm-starting-the-encoder-decoder-model
    """

    # Tokenize the inputs and labels
    inputs = tokenizer_input(batch["orig"])
    outputs = tokenizer_output(batch["norm"])

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # Make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer_output.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch


def load_tokenizers(
    configs: Dict[str, Any]
) -> Tuple[transformers.PreTrainedTokenizerBase, transformers.PreTrainedTokenizerBase]:
    """Load two tokenizer objects based on config dictionary and possibly replace the input tokenizer's normalization component with a custom transliterator"""

    # (A) If tokenizer is given explicitly in config file
    # Load tokenizers
    if "tokenizer_input" in configs["tokenizer"]:
        # Load input tokenizer
        tokenizer_input = transformers.AutoTokenizer.from_pretrained(
            configs["tokenizer"]["tokenizer_input"]
        )
        if "tokenizer_output" in configs["tokenizer"]:
            # Load output tokenizer
            tokenizer_output = transformers.AutoTokenizer.from_pretrained(
                configs["tokenizer"]["tokenizer_output"]
            )
        else:
            # Output tokenizer is simply a reference to input tok
            tokenizer_output = tokenizer_input
        return tokenizer_input, tokenizer_output

    # (B) tokenizer not explicitly stated in config file, but via language model
    # (1) Load tokenizers
    if "checkpoint_encoder_decoder" in configs["language_models"]:
        tokenizer_input = transformers.AutoTokenizer.from_pretrained(
            configs["language_models"]["checkpoint_encoder_decoder"]
        )
        # Output tokenizer is simply a reference to input tok
        tokenizer_output = tokenizer_input
    else:
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
        # (1.3) Load output tokenizer
        tokenizer_output = transformers.AutoTokenizer.from_pretrained(
            configs["language_models"]["checkpoint_decoder"]
        )
    return tokenizer_input, tokenizer_output


def tokenize_dataset_dict(
    dataset_dict: datasets.DatasetDict,
    tokenizer_input: transformers.PreTrainedTokenizerBase,
    tokenizer_output: transformers.PreTrainedTokenizerBase,
    configs,
) -> datasets.DatasetDict:
    """
    Tokenize the datasets in a DatasetDict as specified in the config file.

    Also returns the loaded input and output tokenizers.
    """

    tokenization_kwargs = {
        "tokenizer_input": tokenizer_input,
        "tokenizer_output": tokenizer_output,
    }

    # Tokenize by applying map function to the DatasetDict
    prepared_dataset_dict = dataset_dict.map(
        tokenize_input_and_output,
        fn_kwargs=tokenization_kwargs,
        # remove_columns=["orig", "norm"],
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
        ds = loader.merge_datasets(ds2num_examples, seed=configs["random_seed"])
        ds_split_merged[split] = ds

    dataset = ds_split_merged

    # TODO: is this depcrecated?
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


def warmstart_seq2seq_model(
    configs: Dict[str, Any],
    tokenizer_output: transformers.PreTrainedTokenizerBase,
    device: torch.device,
) -> transformers.PreTrainedModel:
    """
    Load and configure an encoder-decoder model.
    """

    if "checkpoint_encoder_decoder" in configs["language_models"]:
        # TODO: the following is hacky because it only allows a T5-model as encoder_decoder
        model = transformers.T5ForConditionalGeneration.from_pretrained(
            configs["language_models"]["checkpoint_encoder_decoder"],
        ).to(device)
    else:
        model = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(
            configs["language_models"]["checkpoint_encoder"],
            configs["language_models"]["checkpoint_decoder"],
        ).to(device)

    # Setting the special tokens
    if "checkpoint_encoder_decoder" in configs["language_models"]:
        # TODO: the following is hacky because it only allows a T5-model as encoder_decoder
        model.config.decoder_start_token_id = tokenizer_output.pad_token_id
        model.config.eos_token_id = tokenizer_output.eos_token_id
        model.config.pad_token_id = tokenizer_output.pad_token_id
    else:
        model.config.decoder_start_token_id = tokenizer_output.cls_token_id
        model.config.eos_token_id = tokenizer_output.sep_token_id
        model.config.pad_token_id = tokenizer_output.pad_token_id

    # Params for beam search decoding
    # model.config.max_length = configs["tokenizer"].get("max_length_output")
    model.config.no_repeat_ngram_size = configs["beam_search_decoding"][
        "no_repeat_ngram_size"
    ]
    model.config.early_stopping = configs["beam_search_decoding"]["early_stopping"]
    model.config.length_penalty = configs["beam_search_decoding"]["length_penalty"]
    model.config.num_beams = configs["beam_search_decoding"]["num_beams"]

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
        evaluation_strategy=configs["training_hyperparams"]["eval_strategy"],
        fp16=configs["training_hyperparams"]["fp16"],
        eval_steps=configs["training_hyperparams"]["eval_steps"],
        num_train_epochs=configs["training_hyperparams"]["epochs"],
        per_device_train_batch_size=configs["training_hyperparams"]["batch_size"],
        per_device_eval_batch_size=configs["training_hyperparams"]["batch_size"],
        save_steps=configs["training_hyperparams"]["save_steps"],
        logging_steps=configs["training_hyperparams"]["logging_steps"],
        group_by_length=True,
    )

    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, padding=configs["tokenizer"]["padding"]
    )

    # TODO: Should this class stay inside the function or not?
    class Seq2SeqTrainerWithCustomLogging(transformers.Seq2SeqTrainer):
        """Subclass of Seq2SeqTrainer with custom log function
        to save loss scores directly during training.
        """

        def log(self, logs: Dict[str, float]) -> None:
            """
            Custom behavior: Additionally write the logs to the output file
            history.log in the model directory
            """
            if self.state.epoch is not None:
                logs["epoch"] = round(self.state.epoch, 2)

            output = {**logs, **{"step": self.state.global_step}}
            self.state.log_history.append(output)
            # start of custom behavior
            # TODO: Filepath to the log file should be specified elsewhere - what's best?
            logfile = os.path.join(
                output_dir, "history.log"
            )  # output_dir comes from the superordinate function
            with open(logfile, "a") as f:
                f.write(json.dumps(output, indent=4) + "\n")
            # end of custom behavior
            self.control = self.callback_handler.on_log(
                self.args, self.state, self.control, logs
            )

    # Instantiate trainer
    trainer = Seq2SeqTrainerWithCustomLogging(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Run training
    trainer.train()

    return None


ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../..")
)
# TODO pass this on the command-line
CONFIGFILE = os.path.join(ROOT, "training_config.toml")


if __name__ == "__main__":
    # (1) Preparations
    # Load configs
    with open(CONFIGFILE, mode="rb") as fp:
        CONFIGS = tomli.load(fp)
    MODELDIR = os.path.join(ROOT, "./models/model")

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

    print("Loading the data ...")
    dataset_dict = load_and_merge_datasets(CONFIGS)

    # (3) Tokenize data
    print("Tokenizing and preparing the data ...")
    tokenizer_input, tokenizer_output = load_tokenizers(CONFIGS)
    prepared_dataset_dict = tokenize_dataset_dict(
        dataset_dict, tokenizer_input, tokenizer_output, CONFIGS
    )

    # (3.1) Optional: Filter data for length
    prepared_dataset_dict = filter_dataset_dict_for_length(
        prepared_dataset_dict, CONFIGS
    )

    # (4) Load models

    print("Loading the pre-trained models ...")
    model = warmstart_seq2seq_model(CONFIGS, tokenizer_output, device)

    # (5) Training

    print("Training model ...")
    train_seq2seq_model(
        model,
        prepared_dataset_dict["train"],
        prepared_dataset_dict["validation"],
        tokenizer_input,
        CONFIGS,
        MODELDIR,
    )

    # (6) Saving the final model

    model_path = os.path.join(MODELDIR, "model_final/")
    model.save_pretrained(model_path)

    # (7) Save the config file to model directory
    shutil.copy(CONFIGFILE, MODELDIR)
