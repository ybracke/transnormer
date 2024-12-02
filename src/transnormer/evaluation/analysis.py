from typing import List
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def get_spans_of_unknown_tokens(
    text: str, tokenizer: PreTrainedTokenizerBase
) -> List[str]:
    """
    For a given string and tokenizer return the tokens that are unknown to the
    tokenizer

    This only works with huggingface "fast" tokenizers:
    https://huggingface.co/docs/tokenizers
    https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerFast
    """

    encoding = tokenizer(text)
    spans_unk_tokens = [
        # map a token index to a pair of character indices
        encoding.token_to_chars(token_index)[:]
        # we are only interested in the token indices of unknown tokens
        for token_index, token_id in enumerate(encoding["input_ids"])
        if token_id == tokenizer.unk_token_id
    ]
    return spans_unk_tokens
