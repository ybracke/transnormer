import copy
import logging
import os
import pytest

from typing import Any, Dict

import datasets
import transformers
import torch

from transnormer.models import train_model
from transnormer.data import loader

# logging.basicConfig(
#     filename='tests/testlogs/test.log',
#     level=logging.INFO,
#     format='%(asctime)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )

# disable caching in datasets
datasets.disable_caching()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
LOGGER = logging.getLogger(__name__)


def test_tokenize_input_and_output_single_tokenizer() -> None:
    # Load data
    path = "./tests/testdata/dtaeval/txt/arnima_invalide_1818-head10.txt"
    dataset = loader.load_dtaeval_as_dataset(path)

    # Load tokenizers
    tokenizer_input = transformers.AutoTokenizer.from_pretrained("google/byt5-small")

    # Parameters for tokenizing input and labels
    fn_kwargs = {
        "tokenizer": tokenizer_input,
    }

    # Do tokenization via mapping
    prepared_dataset = dataset.map(
        train_model.tokenize_input_and_output,
        fn_kwargs=fn_kwargs,
        batched=True,
        batch_size=1,
        remove_columns=["orig", "norm"],
    )

    assert len(prepared_dataset[0]["input_ids"]) == 58
    assert len(prepared_dataset[0]["attention_mask"]) == 58
    assert len(prepared_dataset[0]["labels"]) == 56


def test_tokenize_dataset_dict_single_tokenizer() -> None:
    CONFIGS: Dict[str, Any] = {
        "gpu": "cuda:0",
        "random_seed": 42,
        "data": {
            "paths_train": [
                "tests/testdata/jsonl/dtaeval-train-head3.jsonl",
                "tests/testdata/jsonl/dtak-1600-1699-train-head3.jsonl",
            ],
            "paths_validation": [
                "tests/testdata/jsonl/dtaeval-train-head3.jsonl",
                "tests/testdata/jsonl/dtak-1600-1699-train-head3.jsonl",
            ],
            "paths_test": [
                "tests/testdata/jsonl/dtaeval-train-head3.jsonl",
                "tests/testdata/jsonl/dtak-1600-1699-train-head3.jsonl",
            ],
            "n_examples_train": [
                1_000_000,
                1_000_000,
            ],
            "n_examples_validation": [
                1_000_000,
                1_000_000,
            ],
            "n_examples_test": [
                1_000_000,
                1_000_000,
            ],
        },
        "subset_sizes": {"train": 3, "validation": 2, "test": 1},
        "tokenizer": {
            #     "max_length_input": 512,
            #     "max_length_output": 512,
        },
        "language_models": {"checkpoint_encoder_decoder": "google/byt5-small"},
        "training_hyperparams": {
            "batch_size": 10,
        },
    }

    dataset_dict = train_model.load_and_merge_datasets(CONFIGS)
    tokenizer = train_model.load_tokenizer(CONFIGS)
    prepared_dataset_dict = train_model.tokenize_dataset_dict(
        dataset_dict, tokenizer, CONFIGS
    )
    # Check if input_ids have desired type (torch.Tensor)
    assert all(
        isinstance(prepared_dataset_dict["train"]["input_ids"][i], torch.Tensor)
        for i in range(prepared_dataset_dict["train"].num_rows)
    )
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerBase)


def test_filter_dataset_dict_for_length() -> None:
    CONFIGS: Dict[str, Any] = {
        "gpu": "cuda:0",
        "random_seed": 42,
        "data": {
            "paths_train": [
                "tests/testdata/jsonl/dtaeval-train-head3.jsonl",
            ],
            "paths_validation": [
                "tests/testdata/jsonl/dtaeval-train-head3.jsonl",
            ],
            "paths_test": [
                "tests/testdata/jsonl/dtaeval-train-head3.jsonl",
            ],
            "n_examples_train": [
                1_000_000,
            ],
            "n_examples_validation": [
                1_000_000,
            ],
            "n_examples_test": [
                1,
            ],
        },
        "tokenizer": {
            "min_length_input": 40,
            # "max_length_input": 120,
        },
        "language_models": {"checkpoint_encoder_decoder": "google/byt5-small"},
        "training_hyperparams": {
            "batch_size": 10,
        },
    }
    dataset_dict = train_model.load_and_merge_datasets(CONFIGS)
    tokenizer = train_model.load_tokenizer(CONFIGS)
    dataset_dict = train_model.tokenize_dataset_dict(dataset_dict, tokenizer, CONFIGS)
    filtered_dataset_dict = train_model.filter_dataset_dict_for_length(
        dataset_dict, CONFIGS
    )
    assert len(filtered_dataset_dict["train"]) == 2

    # set an additional max_length_input
    CONFIGS["tokenizer"]["max_length_input"] = 120
    dataset_dict = train_model.load_and_merge_datasets(CONFIGS)
    tokenizer = train_model.load_tokenizer(CONFIGS)
    dataset_dict = train_model.tokenize_dataset_dict(dataset_dict, tokenizer, CONFIGS)
    filtered_dataset_dict = train_model.filter_dataset_dict_for_length(
        dataset_dict, CONFIGS
    )
    assert len(filtered_dataset_dict["train"]) == 1


def test_load_and_merge_datasets_from_files_full_sets() -> None:
    CONFIGS: Dict[str, Any] = {
        "gpu": "cuda:0",
        "random_seed": 42,
        "data": {
            "paths_train": [
                "tests/testdata/jsonl/dtaeval-train-head3.jsonl",
                "tests/testdata/jsonl/dtak-1600-1699-train-head3.jsonl",
            ],
            "paths_validation": [
                "tests/testdata/jsonl/dtaeval-train-head3.jsonl",
                "tests/testdata/jsonl/dtak-1600-1699-train-head3.jsonl",
            ],
            "paths_test": [
                "tests/testdata/jsonl/dtaeval-train-head3.jsonl",
                "tests/testdata/jsonl/dtak-1600-1699-train-head3.jsonl",
            ],
            "n_examples_train": [
                1_000_000,
                1_000_000,
            ],
            "n_examples_validation": [
                1_000_000,
                1_000_000,
            ],
            "n_examples_test": [
                1_000_000,
                1_000_000,
            ],
        },
        # The rest of the configs doesn't matter ...
    }

    dataset = train_model.load_and_merge_datasets(CONFIGS)
    assert dataset["train"].num_rows == 6
    assert dataset["validation"].num_rows == 6
    assert dataset["test"].num_rows == 6


def test_load_and_merge_datasets_from_files_subsets1() -> None:
    CONFIGS: Dict[str, Any] = {
        "gpu": "cuda:0",
        "random_seed": 42,
        "data": {
            "paths_train": [
                "tests/testdata/jsonl/dtaeval-train-head3.jsonl",
                "tests/testdata/jsonl/dtak-1600-1699-train-head3.jsonl",
            ],
            "paths_validation": [
                "tests/testdata/jsonl/dtaeval-train-head3.jsonl",
                "tests/testdata/jsonl/dtak-1600-1699-train-head3.jsonl",
            ],
            "paths_test": [
                "tests/testdata/jsonl/dtaeval-train-head3.jsonl",
                "tests/testdata/jsonl/dtak-1600-1699-train-head3.jsonl",
            ],
            "n_examples_train": [
                3,
                3,
            ],
            "n_examples_validation": [
                2,
                2,
            ],
            "n_examples_test": [
                1,
                1,
            ],
        },
        # The rest of the configs doesn't matter ...
    }

    dataset = train_model.load_and_merge_datasets(CONFIGS)
    assert dataset["train"].num_rows == 6
    assert dataset["validation"].num_rows == 4
    assert dataset["test"].num_rows == 2


def test_load_and_merge_datasets_from_directories_full_sets() -> None:
    CONFIGS: Dict[str, Any] = {
        "gpu": "cuda:0",
        "random_seed": 42,
        "data": {
            "paths_train": [
                "tests/testdata/jsonl/dir1/identical.jsonl",
                "tests/testdata/jsonl/dir1/reverse.jsonl",
                "tests/testdata/jsonl/dir2/dtaeval-train-head3.jsonl",
            ],
            "paths_validation": [
                "tests/testdata/jsonl/dir1/identical.jsonl",
                "tests/testdata/jsonl/dir1/reverse.jsonl",
            ],
            "paths_test": [
                "tests/testdata/jsonl/dir2/dtaeval-train-head3.jsonl",
            ],
            "n_examples_train": [
                1_000_000_000,
                1_000_000_000,
                1_000_000_000,
            ],
            "n_examples_validation": [
                100000,
                100000,
            ],
            "n_examples_test": [
                1_000_000_000,
            ],
        },
        # The rest of the configs doesn't matter ...
    }
    dataset = train_model.load_and_merge_datasets(CONFIGS)

    CONFIGS2: Dict[str, Any] = {
        "gpu": "cuda:0",
        "random_seed": 42,
        "data": {
            "paths_train": [
                "tests/testdata/jsonl/dir1",
                "tests/testdata/jsonl/dir2",
            ],
            "paths_validation": [
                "tests/testdata/jsonl/dir1",
            ],
            "paths_test": [
                "tests/testdata/jsonl/dir2",
            ],
            "n_examples_train": [
                1_000_000_000,
                1_000_000_000,
            ],
            "n_examples_validation": [
                100000,
            ],
            "n_examples_test": [
                1_000_000_000,
            ],
        },
        # The rest of the configs doesn't matter ...
    }
    dataset2 = train_model.load_and_merge_datasets(CONFIGS2)

    # assert equality of datasets' contents
    assert (
        dataset["validation"].sort("norm")[:] == dataset2["validation"].sort("norm")[:]
    )
    assert dataset["test"].sort("norm")[:] == dataset2["test"].sort("norm")[:]
    assert dataset["train"].sort("norm")[:] == dataset2["train"].sort("norm")[:]


def test_processing_trainset_from_directories() -> None:
    CONFIGS: Dict[str, Any] = {
        "gpu": "cuda:0",
        "random_seed": 42,
        "data": {
            "paths_train": [
                "tests/testdata/jsonl/dir1/identical.jsonl",
                "tests/testdata/jsonl/dir1/reverse.jsonl",
                "tests/testdata/jsonl/dir2/dtaeval-train-head3.jsonl",
            ],
            "paths_validation": [
                "tests/testdata/jsonl/dir1/identical.jsonl",
                "tests/testdata/jsonl/dir1/reverse.jsonl",
            ],
            "paths_test": [
                "tests/testdata/jsonl/dir2/dtaeval-train-head3.jsonl",
            ],
            "n_examples_train": [
                1_000_000_000,
                1_000_000_000,
                1_000_000_000,
            ],
            "n_examples_validation": [
                100000,
                100000,
            ],
            "n_examples_test": [
                1_000_000_000,
            ],
        },
        "tokenizer": {
            "min_length_input": 0,
            "max_length_input": 120,
            "padding": "longest",
        },
        "language_models": {"checkpoint_encoder_decoder": "google/byt5-small"},
        "training_hyperparams": {
            "batch_size": 10,
        },
        # The rest of the configs doesn't matter ...
    }

    CONFIGS2: Dict[str, Any] = {
        "gpu": "cuda:0",
        "random_seed": 42,
        "data": {
            "paths_train": [
                "tests/testdata/jsonl/dir1",
                "tests/testdata/jsonl/dir2",
            ],
            "paths_validation": [
                "tests/testdata/jsonl/dir1",
            ],
            "paths_test": [
                "tests/testdata/jsonl/dir2",
            ],
            "n_examples_train": [
                1_000_000_000,
                1_000_000_000,
            ],
            "n_examples_validation": [
                100000,
            ],
            "n_examples_test": [
                1_000_000_000,
            ],
        },
        "tokenizer": {
            "min_length_input": 0,
            "max_length_input": 120,
            "padding": "longest",
        },
        "language_models": {"checkpoint_encoder_decoder": "google/byt5-small"},
        "training_hyperparams": {
            "batch_size": 10,
        },
        # The rest of the configs doesn't matter ...
    }
    dataset_dict = train_model.load_and_merge_datasets(CONFIGS)
    dataset_dict2 = train_model.load_and_merge_datasets(CONFIGS2)
    tokenizer = train_model.load_tokenizer(CONFIGS)
    tokenizer2 = train_model.load_tokenizer(CONFIGS2)
    dataset_dict = train_model.tokenize_dataset_dict(dataset_dict, tokenizer, CONFIGS)
    dataset_dict2 = train_model.tokenize_dataset_dict(
        dataset_dict2, tokenizer2, CONFIGS2
    )
    prepared_dataset_dict = train_model.filter_dataset_dict_for_length(
        dataset_dict, CONFIGS
    )
    prepared_dataset_dict2 = train_model.filter_dataset_dict_for_length(
        dataset_dict2, CONFIGS2
    )
    # print(prepared_dataset_dict["validation"].sort("norm")[:])
    # print(prepared_dataset_dict2["validation"].sort("norm")[:])

    # Datasets have the same content
    assert str(prepared_dataset_dict["validation"].sort("norm")[:]) == str(
        prepared_dataset_dict2["validation"].sort("norm")[:]
    )
    assert str(prepared_dataset_dict["test"].sort("norm")[:]) == str(
        prepared_dataset_dict2["test"].sort("norm")[:]
    )
    assert str(prepared_dataset_dict["train"].sort("norm")[:]) == str(
        prepared_dataset_dict2["train"].sort("norm")[:]
    )


def test_warmstart_seq2seq_model_single_encoder_decoder() -> None:
    CONFIGS: Dict[str, Any] = {
        "gpu": "cuda:0",
        "random_seed": 42,
        "data": {
            "paths_train": [
                "tests/testdata/jsonl/dtaeval-train-head3.jsonl",
                "tests/testdata/jsonl/dtak-1600-1699-train-head3.jsonl",
            ],
            "paths_validation": [
                "tests/testdata/jsonl/dtaeval-train-head3.jsonl",
                "tests/testdata/jsonl/dtak-1600-1699-train-head3.jsonl",
            ],
            "paths_test": [
                "tests/testdata/jsonl/dtaeval-train-head3.jsonl",
                "tests/testdata/jsonl/dtak-1600-1699-train-head3.jsonl",
            ],
            "n_examples_train": [
                3,
                3,
            ],
            "n_examples_validation": [
                2,
                2,
            ],
            "n_examples_test": [
                1,
                1,
            ],
        },
        "tokenizer": {"padding": "longest"},
        "language_models": {
            "checkpoint_encoder_decoder": "google/byt5-small",
        },
        "training_hyperparams": {
            "batch_size": 10,
            "epochs": 10,
            "eval_steps": 1000,
            "eval_strategy": "steps",
            "save_steps": 10,
            "fp16": False,
        },
    }
    gpu_index = CONFIGS.get("gpu")
    device = torch.device(
        gpu_index if gpu_index is not None and torch.cuda.is_available() else "cpu"
    )
    tokenizer = train_model.load_tokenizer(CONFIGS)
    model = train_model.warmstart_seq2seq_model(CONFIGS, tokenizer, device)
    # Check class
    assert isinstance(model, transformers.T5ForConditionalGeneration)


def test_data_collation() -> None:
    BATCH_SIZE = 8
    CONFIGS: Dict[str, Any] = {
        "gpu": "cuda:0",
        "random_seed": 42,
        "data": {
            "paths_train": [
                "tests/testdata/jsonl/dtaeval-train-16.jsonl",
            ],
            "paths_validation": [
                "tests/testdata/jsonl/dtaeval-train-16.jsonl",
            ],
            "paths_test": [
                "tests/testdata/jsonl/dtaeval-train-head3.jsonl",
            ],
            "n_examples_train": [
                16,
            ],
            "n_examples_validation": [
                16,
            ],
            "n_examples_test": [
                1,
            ],
        },
        "tokenizer": {
            "padding": "longest",
        },
        "language_models": {
            "checkpoint_encoder_decoder": "google/byt5-small",
        },
        "training_hyperparams": {
            "batch_size": BATCH_SIZE,
            "epochs": 1,
            "logging_steps": 1,
            "eval_steps": 1,
            "eval_strategy": "steps",
            "save_steps": 1,
            "fp16": False,  # set to False for byT5-based models
        },
    }

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    gpu_index = CONFIGS.get("gpu")
    device = torch.device(
        gpu_index if gpu_index is not None and torch.cuda.is_available() else "cpu"
    )

    dataset_dict = train_model.load_and_merge_datasets(CONFIGS)
    tokenizer = train_model.load_tokenizer(CONFIGS)
    prepared_dataset_dict = train_model.tokenize_dataset_dict(
        dataset_dict, tokenizer, CONFIGS
    )
    model = train_model.warmstart_seq2seq_model(CONFIGS, tokenizer, device)

    output_dir = "tests/testdata/tmp"

    train_dataset = prepared_dataset_dict["train"]
    eval_dataset = prepared_dataset_dict["validation"]

    # Set-up training arguments from hyperparameters
    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=output_dir,
        predict_with_generate=True,
        evaluation_strategy=CONFIGS["training_hyperparams"]["eval_strategy"],
        fp16=CONFIGS["training_hyperparams"]["fp16"],
        eval_steps=CONFIGS["training_hyperparams"]["eval_steps"],
        num_train_epochs=CONFIGS["training_hyperparams"]["epochs"],
        per_device_train_batch_size=CONFIGS["training_hyperparams"]["batch_size"],
        per_device_eval_batch_size=CONFIGS["training_hyperparams"]["batch_size"],
        save_steps=CONFIGS["training_hyperparams"]["save_steps"],
        logging_steps=CONFIGS["training_hyperparams"]["logging_steps"],
        group_by_length=True,
        # remove_unused_columns=False, # for test reasons - doesn't work
    )

    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        padding=CONFIGS["tokenizer"]["padding"],  # "longest"
    )

    # Instantiate trainer
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Get batches
    for i, batch in enumerate(trainer.get_train_dataloader()):
        # assert that all input_ids/labels are padded to length of the longest example
        length_longest_input_in_batch = max(
            [len([t for t in s if t != 0]) for s in batch["input_ids"]]
        )
        assert batch["input_ids"].shape == (BATCH_SIZE, length_longest_input_in_batch)
        length_longest_output_in_batch = max(
            [len([t for t in s if t != -100]) for s in batch["labels"]]
        )
        assert batch["labels"].shape == (BATCH_SIZE, length_longest_output_in_batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="training requires cuda")
def test_train_seq2seq_model_single_encoder_decoder() -> None:
    CONFIGS: Dict = {
        "gpu": "cuda:0",
        "random_seed": 42,
        "data": {
            "paths_train": [
                "tests/testdata/jsonl/ascii-reverse.jsonl",
                "tests/testdata/jsonl/ascii-reverse.jsonl",
                # "tests/testdata/jsonl/dtak-1600-1699-train-head3.jsonl",
            ],
            "paths_validation": [
                "tests/testdata/jsonl/ascii-reverse.jsonl",
                # "tests/testdata/jsonl/dtak-1600-1699-train-head3.jsonl",
            ],
            "paths_test": [
                "tests/testdata/jsonl/ascii-reverse.jsonl",
                # "tests/testdata/jsonl/dtak-1600-1699-train-head3.jsonl",
            ],
            "n_examples_train": [
                1,
                1,
            ],
            "n_examples_validation": [
                1,
            ],
            "n_examples_test": [
                1,
            ],
        },
        "tokenizer": {
            "padding": "longest",
        },
        "language_models": {
            "checkpoint_encoder_decoder": "google/byt5-small",
        },
        "training_hyperparams": {
            "batch_size": 1,
            "epochs": 2,
            "learning_rate": 0.001,
            "logging_steps": 1000,
            "eval_steps": 1000,
            "save_strategy": "epoch",
            "eval_strategy": "epoch",
            "logging_strategy": "epoch",
            "fp16": False,  # set to False for byT5-based models
            "save_total_limit": 1,
        },
    }
    gpu_index = CONFIGS.get("gpu")
    device = torch.device(
        gpu_index if gpu_index is not None and torch.cuda.is_available() else "cpu"
    )
    dataset_dict = train_model.load_and_merge_datasets(CONFIGS)
    tokenizer = train_model.load_tokenizer(CONFIGS)
    prepared_dataset_dict = train_model.tokenize_dataset_dict(
        dataset_dict, tokenizer, CONFIGS
    )
    model = train_model.warmstart_seq2seq_model(CONFIGS, tokenizer, device)

    model_untrained = copy.deepcopy(model)
    output_dir = "tests/testdata/tmp"

    # Training
    train_dataset = prepared_dataset_dict["train"]
    eval_dataset = prepared_dataset_dict["validation"]
    train_model.train_seq2seq_model(
        model, train_dataset, eval_dataset, tokenizer, CONFIGS, output_dir
    )
    # Compare all states and check that some of them changed
    unequal_states = []
    for (name_mo, params_mo), (name_mn, params_mn) in zip(
        model_untrained.state_dict().items(), model.state_dict().items()
    ):
        assert name_mo == name_mn
        if not torch.equal(params_mo, params_mn):
            unequal_states.append(name_mo)
    assert len(unequal_states) > 0
    # Remove files that were created during training
    for root, dirs, files in os.walk(output_dir, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        else:
            os.rmdir(root)
