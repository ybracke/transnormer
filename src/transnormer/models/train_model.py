from datetime import datetime
import logging
import os
import random
import time
import tomli
import tracemalloc
from typing import Tuple

# import codecarbon
import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
import transformers

from transformers import AutoTokenizer
from tokenizers.normalizers import Normalizer
from tokenizers import NormalizedString

from transnormer.data.loader import load_dtaevalxml_all

# TODO pass this on the command-line
ROOT = "/home/bracke/code/transnormer"
CONFIGFILE = os.path.join(ROOT, "training_config.toml")


# not used yet
def train(
    model: transformers.BertForMaskedLM,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Training loop"""

    return None


class CustomNormalizer:
    def normalize(self, normalized: NormalizedString):
        # Decompose combining characters
        normalized.nfd()  
        # Some character conversions
        normalized.replace("ſ", "s")
        normalized.replace("ꝛ", "r")
        normalized.replace(chr(0x0303), "")  # drop combining tilde
        normalized.replace(chr(0x0364), "e") # convert "Combining Latin Small Letter E" to "e"
        normalized.replace("æ", "ae")
        normalized.replace("ů", "ü")
        normalized.replace("Ů", "Ü")
        # Unicode composition (put decomposed chars back together)
        normalized.nfc()


if __name__ == "__main__":

    # Load configs
    with open(CONFIGFILE, mode="rb") as fp:
        configs = tomli.load(fp)

    # Fix seeds for reproducibilty
    random.seed(configs["random_seed"])
    np.random.seed(configs["random_seed"])
    torch.manual_seed(configs["random_seed"])

    # Start tracking time, memory, emissions
    start_all = time.time()
    tracemalloc.start()
    # tracker.start()

    # GPU set-up
    device = torch.device(configs["gpu"] if torch.cuda.is_available() else "cpu")


    ##################  Load data ###########################
    print("Loading the data ...")

    # TODO: Should the path be hard-coded?
    DATADIR = "/home/bracke/data/dta/dtaeval/split-v3.0/xml"
    dta_dataset = load_dtaevalxml_all(DATADIR, filter_classes=["BUG", "FM", "GRAPH"])

    # Create smaller datasets from random examples
    if "subset_sizes" in configs:
        train_size = configs["subset_sizes"]["train"]
        validation_size = configs["subset_sizes"]["validation"]
        test_size = configs["subset_sizes"]["test"]
        dta_dataset["train"] = dta_dataset["train"].shuffle().select(range(train_size))
        dta_dataset["validation"] = (
            dta_dataset["validation"].shuffle().select(range(validation_size))
        )
        dta_dataset["test"] = dta_dataset["test"].shuffle().select(range(test_size))


    # ################## Tokenization #########################
    print("Tokenizing and preparing the data ...")
    start = time.process_time()

    # Load tokenizers
    tokenizer_hmbert_custom = AutoTokenizer.from_pretrained(
        "dbmdz/bert-base-historic-multilingual-cased"
    )
    tokenizer_hmbert_custom.backend_tokenizer.normalizer = Normalizer.custom(
        CustomNormalizer()
    )

    tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    # TODO: Maximum tokens for input and output; these values are picked randomly
    encoder_max_length = 128
    decoder_max_length = 128


    # This is from here: 
    # https://huggingface.co/blog/warm-starting-encoder-decoder#warm-starting-the-encoder-decoder-model
    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        inputs = tokenizer_hmbert_custom(
            batch["orig"],
            padding="max_length",
            truncation=True,
            max_length=encoder_max_length,
        )
        outputs = tokenizer_bert(
            batch["norm"],
            padding="max_length",
            truncation=True,
            max_length=decoder_max_length,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [
                -100 if token == tokenizer_bert.pad_token_id else token
                for token in labels
            ]
            for labels in batch["labels"]
        ]

        return batch

    # Apply the mapping above
    prepared_dataset = dta_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=configs["training_hyperparams"]["batch_size"],
        remove_columns=["orig", "norm"],
    )

    # Convert to torch tensors
    prepared_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    end = time.process_time()
    print(f"Elapsed time for tokenization: {end - start}")


    # ################## Warm-start the model #################

    print("Loading the pre-trained models ...")

    model = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(
        configs["language_models"]["checkpoint_encoder"],
        configs["language_models"]["checkpoint_decoder"],
    ).to(device)

    # The following two sections were just copied from here:
    # https://huggingface.co/blog/warm-starting-encoder-decoder#warm-starting-the-encoder-decoder-model
    # Setting the special tokens
    model.config.decoder_start_token_id = tokenizer_bert.cls_token_id
    model.config.eos_token_id = tokenizer_bert.sep_token_id
    model.config.pad_token_id = tokenizer_hmbert_custom.pad_token_id # 0
    # model.config.vocab_size = model.config.encoder.vocab_size

    # Params for beam search decoding (see https://huggingface.co/blog/how-to-generate)
    model.config.max_length = 128
    # model.config.min_length = 56
    model.config.no_repeat_ngram_size = 3 # ???
    model.config.early_stopping = True
    model.config.length_penalty = 2.0
    model.config.num_beams = 4


    # ################## Training #############################
    print("Training ...")


    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=os.path.join(ROOT,f"./models/models_{timestamp}/"),
        predict_with_generate=True,
        evaluation_strategy="steps",
        fp16=True,
        eval_steps=1000,
        num_train_epochs=configs["training_hyperparams"]["epochs"],
        per_device_train_batch_size=configs["training_hyperparams"]["batch_size"],
        per_device_eval_batch_size=configs["training_hyperparams"]["batch_size"],
        # logging_steps=2,
        # save_steps defaults to 500 -> TODO increase this
        # save_steps=5000, # ? doesn't work because a custom tokenizer can't be saved
    )

    # instantiate trainer
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        # tokenizer=tokenizer_hmbert_custom,
        args=training_args,
        train_dataset=prepared_dataset["train"],
        eval_dataset=prepared_dataset["validation"],
        # compute_metrics=compute_metrics, # TODO
    )

    trainer.train()

    #### Saving the model
    ## this works
    model_path = os.path.join(ROOT,f"./models/models_{timestamp}/model_final/")
    model.save_pretrained(model_path)
    ## this fails because a custom tokenizer can't be saved
    # model_path = f"./models/model_fromtrainer/"
    # trainer.save_model(model_path)



