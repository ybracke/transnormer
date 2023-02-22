import datasets
from transformers import AutoTokenizer

from transnormer.models.train_model import tokenize_input_and_output
from transnormer.data.loader import *

import tomli

def test_tokenize_input_and_output():
    
    # Load data
    path = "./tests/testdata/txt/arnima_invalide_1818-head10.txt"
    dataset = load_dtaeval_as_dataset(path)

    # Load tokenizers
    tokenizer_input = AutoTokenizer.from_pretrained(
        "dbmdz/bert-base-historic-multilingual-cased"
    )
    tokenizer_labels = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    # Parameters for tokenizing input and labels
    fn_kwargs = {"tokenizer_input": tokenizer_input, 
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
    
    assert(len(prepared_dataset[0]["input_ids"])==128)
    assert(len(prepared_dataset[0]["attention_mask"])==128)
    assert(len(prepared_dataset[0]["labels"])==128)

def test_parameter_loading():
    target_param_dict = {
        "gpu" : "cuda:0",
        "random_seed" : 42,
        "data" : {
            "path" : "./data/dta/dtaeval/split-v3.1/txt",
        },
        "subset_sizes" : {
            "train" : 100,
            "validation" : 10,
            "test" : 1
        },
        "tokenizer" : {
            "max_length_input" : 128,
            "max_length_output" : 128,
        },
        "language_models" : {
            "checkpoint_encoder" : "dbmdz/bert-base-historic-multilingual-cased",
            "checkpoint_decoder" : "bert-base-multilingual-cased"
        },
        "training_hyperparams" : {
            "batch_size" : 10,
            "epochs" : 10,
            "eval_steps" : 1000,
            "eval_strategy" : "steps",
            "save_steps" : 10,
            "fp16" : True,
        },
        "beam_search_decoding" : {
            "no_repeat_ngram_size" : 3,
            "early_stopping" : True,
            "length_penalty" : 2.0,
            "num_beams" : 4,
            }
    }
    
    path = "training_config_template.toml"
    with open(path, mode="rb") as fp:
        CONFIGS = tomli.load(fp)
    assert CONFIGS == target_param_dict
