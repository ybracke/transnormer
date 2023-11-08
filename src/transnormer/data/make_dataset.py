# -*- coding: utf-8 -*-
import argparse
import os
from typing import Optional, List

import datasets

from transnormer.data.loader import load_data

"""
usage: make_dataset.py [-h] [-t TARGET] DATASET [DATASET ...]

Example call:
python3 src/transnormer/data/make_dataset.py \
    data/raw/ridges/bollmann-split/ridges.dev.txt \
    data/raw/ridges/bollmann-split/ridges.test.txt \
    data/raw/ridges/bollmann-split/ridges.train.txt \
    data/raw/dta/dtaeval/split-v3.0/xml/dev \
    data/raw/dta/dtaeval/split-v3.0/xml/test \
    data/raw/dta/dtaeval/split-v3.0/xml/train \
    data/raw/leipzig-corpora/deu_news_2020_1M-sentences.txt \
    --target data/interim
"""


def save_to_jsonl(paths: List[str], parent_dir: str) -> None:
    """
    Store all datasets (or dataset splits) in `paths` in JSON Lines format
    under `target_dir`
    """
    for name, split, data in load_data(paths):
        ds = datasets.Dataset.from_dict(data)
        target_dir = os.path.join(parent_dir, name)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        if split:
            target_file = os.path.join(target_dir, f"{name}-{split}.jsonl")
        else:
            target_file = os.path.join(target_dir, f"{name}.jsonl")
        ds.to_json(target_file, force_ascii=False)


def parse_arguments(arguments: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert datasets from raw format to JSON Lines"
    )
    parser.add_argument("DATASET", nargs="+", help="Path(s) to dataset(s)")
    parser.add_argument(
        "-t", "--target", default="./data/interim", help="Path to target directory"
    )

    return parser.parse_args(arguments)


def main(arguments: Optional[List[str]] = None) -> None:
    args = parse_arguments(arguments)
    save_to_jsonl(args.DATASET, args.target)


if __name__ == "__main__":
    main()
