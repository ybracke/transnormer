import argparse
import os
from typing import List, Optional
import datasets


"""
Create train, validation and test splits from a dataset (JSON Lines)


usage: split_dataset.py [-h] [-o OUT]
                        [-v VALIDATION_SET_SIZE] [-t TEST_SET_SIZE]
                        [--random-state RANDOM_STATE] file
"""

RANDOM_STATE = 42


def parse_arguments(arguments: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create genre-stratified train, validation and test splits of the DTA for a specific time frame."
    )
    parser.add_argument("file", type=str, help="Input file (JSON Lines).")

    parser.add_argument(
        "-o", "--out", type=str, help="Path to the output file (JSON Lines)."
    )

    parser.add_argument(
        "-v",
        "--validation-set-size",
        type=float,
        default=0.1,
        help="Size of the validation set as a fraction of the total data.",
    )

    parser.add_argument(
        "-t",
        "--test-set-size",
        type=float,
        default=0.1,
        help="Size of the test set as a fraction of the total data.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help=f"Seed for the random state (default: {RANDOM_STATE}).",
    )

    return parser.parse_args(arguments)


def main(arguments: Optional[List[str]] = None) -> None:
    # (1) Read arguments
    args = parse_arguments(arguments)

    # (2) Load the jsonl file
    ds_splits = datasets.load_dataset("json", data_files=args.file)
    ds_splits = ds_splits["train"].train_test_split(
        test_size=args.test_set_size, seed=args.random_state
    )
    val_size = args.validation_set_size / (1.0 - args.test_set_size)
    ds_splits["train"], ds_splits["validation"] = (
        ds_splits["train"]
        .train_test_split(test_size=val_size, seed=args.random_state)
        .values()
    )

    # (3) Save each split
    for split, ds in ds_splits.items():
        if args.out:
            outfile = args.out
        else:
            fname = os.path.splitext(os.path.basename(args.file))[0]
            outfile = os.path.join(os.path.dirname(args.file), f"{fname}-{split}.jsonl")

        if not os.path.dirname(outfile):
            raise ValueError(
                f"Directory {os.path.dirname(outfile)} does not exist. Please create it first."
            )

        # Note: non-ASCII chars are escaped with method ".to_json"
        ds.to_json(outfile)

    return None


if __name__ == "__main__":
    main()
