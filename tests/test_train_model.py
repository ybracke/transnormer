from transformers import AutoTokenizer

from transnormer.models.train_model import tokenize_input_and_output
from transnormer.data import loader

import tomli


def test_tokenize_input_and_output():
    # Load data
    path = "./tests/testdata/dtaeval/txt/arnima_invalide_1818-head10.txt"
    dataset = loader.load_dtaeval_as_dataset(path)

    # Load tokenizers
    tokenizer_input = AutoTokenizer.from_pretrained(
        "dbmdz/bert-base-historic-multilingual-cased"
    )
    tokenizer_labels = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    # Parameters for tokenizing input and labels
    fn_kwargs = {
        "tokenizer_input": tokenizer_input,
        "tokenizer_output": tokenizer_labels,
        "max_length_input": 128,
        "max_length_output": 128,
    }

    # Do tokenization via mapping
    prepared_dataset = dataset.map(
        tokenize_input_and_output,
        fn_kwargs=fn_kwargs,
        batched=True,
        batch_size=1,
        remove_columns=["orig", "norm"],
    )

    assert len(prepared_dataset[0]["input_ids"]) == 128
    assert len(prepared_dataset[0]["attention_mask"]) == 128
    assert len(prepared_dataset[0]["labels"]) == 128


def test_config_file_structure():
    target_param_dict = {
        "gpu": "cuda:0",
        "random_seed": 42,
        "data": {
            "paths_train": [
                "data/interim/dtaeval/dtaeval-train.jsonl",
                "data/interim/deu_news_2020/deu_news_2020-train.jsonl",
            ],
            "paths_validation": [
                "data/interim/dtaeval/dtaeval-validation.jsonl",
                "data/interim/deu_news_2020/deu_news_2020-validation.jsonl",
            ],
            "paths_test": [
                "data/interim/dtaeval/dtaeval-test.jsonl",
                "data/interim/deu_news_2020/deu_news_2020-test.jsonl",
            ],
            "n_examples_train": [
                1_000_000_000,
                50_000,
            ],
            "n_examples_validation": [
                1_000_000_000,
                5_000,
            ],
            "n_examples_test": [
                1_000_000_000,
                5_000,
            ],
        },
        "subset_sizes": {"train": 100, "validation": 10, "test": 1},
        "tokenizer": {
            "max_length_input": 128,
            "max_length_output": 128,
        },
        "language_models": {
            "checkpoint_encoder": "dbmdz/bert-base-historic-multilingual-cased",
            "checkpoint_decoder": "bert-base-multilingual-cased",
        },
        "training_hyperparams": {
            "batch_size": 10,
            "epochs": 10,
            "eval_steps": 1000,
            "eval_strategy": "steps",
            "save_steps": 10,
            "fp16": True,
        },
        "beam_search_decoding": {
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
            "length_penalty": 2.0,
            "num_beams": 4,
        },
    }

    path = "training_config_template.toml"
    with open(path, mode="rb") as fp:
        CONFIGS = tomli.load(fp)
    assert CONFIGS == target_param_dict
