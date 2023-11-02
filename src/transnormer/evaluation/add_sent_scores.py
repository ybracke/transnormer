import argparse
import json
import pickle
import re
from typing import Dict, List, Optional


def read_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, "r") as f:
        for line in f:
            record = json.loads(line)
            data.append(record)
    return data


def write_jsonl(data: List[Dict], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return


def parse_arguments(
    arguments: Optional[List[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write sentence-wise accuracy scores stored SCORES to DATA (jsonl file)"
    )

    parser.add_argument(
        "scores",
        help="Scores file (either pickled (*.pkl) or comma-separated plain-text).",
    )
    parser.add_argument(
        "data",
        help="Data file (JSONL)",
    )
    parser.add_argument(
        "-p",
        "--property",
        type=str,
        default="score",
        help="Name for the property in which the score gets stored (default: 'score')",
    )

    args = parser.parse_args(arguments)
    return args


def main(arguments: Optional[List[str]] = None) -> None:
    args = parse_arguments(arguments)

    # Load score list
    pickled_scores = re.match(r".*.pkl", args.scores)
    if pickled_scores:
        with open(args.scores, "rb") as f:
            scores: List[float] = pickle.load(f)
    else:
        with open(args.scores, "r", encoding="utf-8") as f:
            scores = [float(v) for v in f.readline().split(",")]

    # Load data
    data: List[Dict] = read_jsonl(args.data)
    assert len(data) == len(scores), "DATA and SCORES must be of same length"

    # Update data and write to file
    for record, score in zip(data, scores):
        record[args.property] = score
    write_jsonl(data, args.data)

    return


if __name__ == "__main__":
    main()
