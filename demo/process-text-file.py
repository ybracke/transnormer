#!/usr/bin/env python
# coding: utf-8

import argparse
from typing import List, Optional

import torch
import transformers


def text_chunker(
    text: str,
    symbols: List[str] = [",", " "],
    max_length: int = 512,
    split_margin: int = 50,
):
    """
    Yields chunks of `text` split at specified `symbols`, keeping chunk sizes under `max_length`.
    `split_margin`: the margin before `max_length` where the function tries to split `text`.

    This is a helper function for chunking long inputs and prevent intra-word splitting as much as possible by splitting at unproblematic symbols (e.g. space, comma).
    """

    text_length = len(text)
    if text_length <= max_length:
        yield text
        return

    prev_sym_index = 0
    start = max_length - split_margin
    end = max_length

    while start < text_length:
        sym_index = -1
        for sym in symbols:
            # Find last occurrence in range
            idx = text.rfind(sym, start, min(end, text_length))
            if idx != -1 and (sym_index == -1 or idx > sym_index):
                sym_index = idx

        # No split symbol found, force split at max_length
        if sym_index == -1:
            sym_index = min(end, text_length) - 1

        # Yield chunk
        yield text[prev_sym_index : sym_index + 1]  # noqa: E203

        # Move to the next chunk
        prev_sym_index = sym_index + 1
        start = prev_sym_index + max_length - split_margin
        end = prev_sym_index + max_length

    if prev_sym_index < text_length:
        # Yield the remaining text
        yield text[prev_sym_index:]


def normalize(
    input_text: str,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    gen_cfg: transformers.GenerationConfig,
    split_margin: int = 50,
):
    """
    Normalization that uses a text chunker for long inputs
    """
    # Set split margin to 50 characters
    m = split_margin
    max_length = gen_cfg.max_new_tokens
    full_response = []
    for chunk in text_chunker(input_text, max_length=max_length, split_margin=m):
        input_ids = tokenizer(chunk, return_tensors="pt").input_ids
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        outputs = model.generate(
            input_ids,
            generation_config=gen_cfg,
        )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        full_response.append(response_text)

    # Join response
    final_response = " ".join(full_response)

    return final_response


def parse_arguments(
    arguments: Optional[List[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generates normalizations with a transnormer model for the paragraphs in `file` and prints normalized paragraphs to stdout."
    )

    parser.add_argument(
        "--model",
        default="ybracke/transnormer-18-19c-beta-v01",
        help="Model name or location (default: %(default)s)",
    )

    parser.add_argument(
        "file",
        help="Path to the input file (raw text)",
    )

    args = parser.parse_args(arguments)

    return args


def main(arguments: Optional[List[str]] = None) -> None:
    args = parse_arguments(arguments)
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    gen_cfg = transformers.GenerationConfig.from_model_config(model.generation_config)

    with open(args.file, encoding="utf-8") as f:
        full_text = f.read()

    sentences = full_text.split("\n")
    for i, sent in enumerate(sentences):
        normalized_text = normalize(sent.strip(), model, tokenizer, gen_cfg=gen_cfg)
        print(normalized_text)

        # Stop after ten sentences because this is just a demo
        if i > 10:
            break


if __name__ == "__main__":
    main()
