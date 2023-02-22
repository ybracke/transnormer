import datasets
from transformers import AutoTokenizer

from transnormer.models.train_model import tokenize_input_and_output
from transnormer.data.loader import *


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
