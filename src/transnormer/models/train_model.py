# imports
from datetime import datetime
from functools import wraps
import glob
import itertools
import logging
import os
import random
import time
import tomli
import tracemalloc
from typing import Iterator, Dict, List, Generator, Tuple, Optional, Any, TextIO

# import codecarbon
import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
import transformers

from transformers import AutoTokenizer
from tokenizers.normalizers import Normalizer
from tokenizers import NormalizedString

# TODO pass this on the command-line
ROOT = "/home/bracke/code/transnormer"
CONFIGFILE = os.path.join(ROOT, "training_config.toml")


# # Tracking time, memory, carbon
# tracker = codecarbon.OfflineEmissionsTracker(
#     country_iso_code="DEU", log_level="warning"
# )

# # Logging
# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
# logging.basicConfig(
#     filename=f"run_{timestamp}.log",
#     level=logging.INFO,
# )


###################### Helper functions #####################


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.process_time()
        result = func(*args, **kwargs)
        end = time.process_time()
        print(f"Elapsed time for `{func.__name__}`: {end - start}")
        return result

    return wrapper


###################### Functions ############################


@timer
def load_dtaeval_all() -> datasets.DatasetDict:

    datadir = "/home/bracke/data/dta/dtaeval/split-v3.1/txt"

    train_path = os.path.join(datadir, "train")
    validation_path = os.path.join(datadir, "dev")
    test_path = os.path.join(datadir, "test")

    ds = datasets.DatasetDict()
    ds["train"] = load_dtaeval_as_dataset(train_path)
    ds["validation"] = load_dtaeval_as_dataset(validation_path)
    ds["test"] = load_dtaeval_as_dataset(test_path)

    return ds

def load_dtaeval_as_dataset(path:str) -> datasets.Dataset:
    """ 
    Load the file(s) under `path` into a datasets.Dataset with columns "orig" and "norm"

    If `path` is a directory name, 
    """
    
    docs = [load_tsv(file, keep_sentences=True) for file in file_gen(path)]
    docs_sent_joined = [[
            [" ".join(sent) for sent in column] for column in doc
        ] for doc in docs]
        
    all_sents_orig, all_sents_norm = [], []
    for doc_orig, doc_norm in docs_sent_joined:
        all_sents_orig.extend([sent for sent in doc_orig])
        all_sents_norm.extend([sent for sent in doc_norm])

    return datasets.Dataset.from_dict({"orig" : all_sents_orig, "norm" : all_sents_norm})

def load_tsv(file_obj, keep_sentences=True):
    """
    Load a corpus in a tab-separated plain text file into lists

    `keep_sentences` : if set to True, empty lines are interpreted
    as sentence breaks. Consecutive empty lines are ignored.
    """

    line = file_obj.readline()
    # Read upto first non-empty line
    while line.isspace():
        line = file_obj.readline()
    # Number of columns in text file
    n_columns = line.strip().count("\t") + 1
    # Initial empty columns with one empty sentence inside
    columns = [[[]] for i in range(n_columns)]
    # Read file
    line_cnt = 0
    sent_cnt = 0
    while line:
        # non-empty line
        if not line.isspace():
            line = line.strip()
            line_split = line.split("\t")

            # Catch/skip ill-formed lines
            if len(line_split) != n_columns:
                print(
                    f"Line {line_cnt+1} does not have length "
                    f"{n_columns} but {len(line_split)} skip line: '{line}'"
                )
            else:
                # build up sentences
                for i in range(n_columns):
                    columns[i][sent_cnt].append(line_split[i])

        # empty line
        else:
            # current sentence empty?
            # then just replace with empty sentence again
            if columns[0][sent_cnt] == []:
                for i in range(n_columns):
                    columns[i][sent_cnt] = []
            # else: move to build next sentence
            else:
                for i in range(n_columns):
                    columns[i].append([])
                sent_cnt += 1

        # Move on
        line = file_obj.readline()
        line_cnt += 1

    # optional: flatten structure
    if not keep_sentences:
        columns = [list(itertools.chain(*col)) for col in columns]

    return columns

def file_gen(path : str) -> Generator[TextIO, None, None]:
    """ Yields file(s) from a path, where path can be file, dir or glob """

    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as file:
            yield file
    elif os.path.isdir(path):
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
                yield file
    else:
        for filename in glob.glob(path):
            with open(filename, 'r', encoding='utf-8') as file:
                yield file




# not used yet
def add_gold_labels(
    example: datasets.formatting.formatting.LazyRow,
) -> datasets.formatting.formatting.LazyRow:
    """
    Additional data augmentation
    Add a column to the datasets: copy of "input_ids" called "labels"
    During training the data collator will mask the "input_ids" and the "labels"
    will serve as the gold labels

    LazyRow behaves like Dict[str, List[Any]]
    """
    example["labels"] = example["input_ids"].copy()
    return example


# not used yet
def train_one_epoch(
    train_dataloader: DataLoader, optimizer: torch.optim.Optimizer
) -> Tuple[float, float]:
    """Train one iteration over the dataset"""

    running_loss = 0.0
    running_loss_total = 0.0
    last_avg_loss_after_n_batches = 0.0

    for i, batch in enumerate(train_dataloader):
        # Move tensors to GPU
        batch = {k: v.to(device) for k, v in batch.items()}

        # Zero the gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs: transformers.modeling_outputs.MaskedLMOutput = model(**batch)

        # Compute the loss and its gradients
        loss = outputs.loss  # tensor of shape (1,)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        running_loss_total += loss.item()

        # Report avg loss after every `m` batches
        m = configs["training_hyperparams"]["report_avg_loss_after_n_batches"]
        if i % m == (m - 1):
            last_avg_loss_after_n_batches = running_loss / m  #
            print(f"  batch {i+1} loss: {last_avg_loss_after_n_batches}")
            running_loss = 0.0

        # Average loss per training batch
        avg_loss = running_loss_total / (i + 1)

    return avg_loss, last_avg_loss_after_n_batches


# not used yet
def train(
    model: transformers.BertForMaskedLM,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Training loop"""

    best_vloss = 1_000_000.0

    # File to track epochs, losses and resource usage
    # (for now just time, #TODO possibly memory, enery)
    with open(os.path.join(run_dir, "stats.csv"), "w", encoding="utf-8") as f:
        f.write(
            "epoch,avg_loss_train,avg_loss_last_train_batches,avg_loss_val,duration\n"
        )
        f.flush()

        for epoch in range(1, configs["training_hyperparams"]["epochs"] + 1):
            start_epoch = time.time()
            print(f"EPOCH {epoch}:")

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            avg_loss, last_avg_loss = train_one_epoch(train_dataloader, optimizer)

            # 2. Performance on the validation set
            # We don't need gradients on to do reporting
            model.train(False)

            running_vloss = 0.0
            for i, vbatch in enumerate(val_dataloader):
                # Move tensors to GPU
                vbatch = {k: v.to(device) for k, v in vbatch.items()}
                outputs = model(**vbatch)
                loss = outputs.loss
                running_vloss += loss.item()

            # Average loss per validation batch
            avg_vloss = running_vloss / (i + 1)
            print(f"LOSS train: {avg_loss}; valid: {avg_vloss}")
            f.write(
                f"{epoch},{avg_loss:.5f},{last_avg_loss:.5f},{avg_vloss:.5f},{time.time() - start_epoch:.2f}\n"
            )
            f.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = os.path.join(run_dir, f"models/model_epoch_{epoch}")
                model.save_pretrained(model_path)

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

    # # Timestamp is used for naming output files/dirs
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    # run_dir = f"runs/run_{timestamp}"
    # if not os.path.exists(run_dir):
    #     os.makedirs(run_dir)

    # GPU set-up
    device = torch.device(configs["gpu"] if torch.cuda.is_available() else "cpu")


    ##################  Load data ###########################
    print("Loading the data ...")

    dta_dataset = load_dtaeval_all()

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

    # # This is from dta-bert.py, if I use native pytorch for the training loop,
    # # this would become relevant again
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=configs["training_hyperparams"]["learning_rate"],
    #     momentum=configs["training_hyperparams"]["momentum"],
    # )
    # train(model, train_dataloader, val_dataloader, optimizer)

    # # Stop time, memory, emissions tracking
    # # tracker.stop()
    # mem_current, mem_peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # print(
    #     f"Memory usage - Current: {mem_current/1024:.2f} MiB | Peak: {mem_peak/1024:.2f} MiB"
    # )
    # end_all = time.time()
    # print(f"Elapsed time (total): {end_all - start_all}")




    # ################## ~ end of training #######################





    # ################## Application #############################

    # This is the same as in apply_transnomer.ipynb
    print("Application")

    def generate_normalization(batch):
        inputs = tokenizer_hmbert_custom(batch["orig"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=128)

        output_str = tokenizer_bert.batch_decode(outputs, skip_special_tokens=True)

        # batch["norm_pred_ids"] = outputs
        batch["norm_pred_str"] = output_str

        return batch

    batch_size = 4  # change to 64 for full evaluation

    results = dta_dataset["validation"].select(range(batch_size)).map(
        generate_normalization,
        batched=True,
        batch_size=batch_size,
    )

    print(results)
