# Custom evaluation script that uses (slightly modified) evaluation scripts
# developed by Bawden et al (2022), see: https://github.com/rbawden/ModFr-Norm#evaluation

import argparse
import json
import re
import pickle
from typing import Any, Dict, List, Optional

from transnormer.evaluation.metrics import word_acc_final as acc
from transnormer.evaluation.metrics import lev_norm_corpuslevel as lev_norm_c
from transnormer.evaluation import tokenise


def read_jsonl_file(file_path: str, field_name: str) -> List[str]:
    data = []
    with open(file_path, "r") as file:
        for line in file:
            record = json.loads(line)
            if field_name in record:
                data.append(record[field_name].strip())
    return data


def read_plain_text_file(file_path: str) -> List[str]:
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(line.strip())
    return data


def get_metrics(
    ref: List[str], pred: List[str], align_types: List[str]
) -> Dict[str, Any]:
    """Computes evaluation metrics over two lists of sentences

    Internally each sentence is tokenized and aligned with its corresponding sentence in the other list. Then, the scores are computed.
    """
    ref_tok = [tokenise.basic_tokenise(sent) for sent in ref]
    pred_tok = [tokenise.basic_tokenise(sent) for sent in pred]

    metrics: Dict[str, Any] = {"n": len(ref)}

    acc_scores = acc(ref_tok, pred_tok, align_types)
    metrics["acc_harmonized"] = acc_scores["both"] if "both" in align_types else None
    metrics["per_sent"] = acc_scores["per_sent"]

    dist_score = lev_norm_c(ref, pred)
    metrics["dist_norm_c"] = dist_score

    return metrics


def parse_and_check_arguments(
    arguments: Optional[List[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute evaluation metric(s) for string-to-string normalization (see Bawden et al. 2022). Choose --align-type=both for a harmonized accuracy score."
    )

    parser.add_argument(
        "--input-type",
        choices=["jsonl", "text"],
        required=True,
        help="Type of input files: jsonl or text",
    )
    parser.add_argument(
        "--ref-file",
        help="Path to the input file containing reference normalizations (typically a gold standard)",
    )
    parser.add_argument(
        "--pred-file", help="Path to the input file containing predicted normalizations"
    )
    parser.add_argument(
        "--ref-field",
        help="Name of the field containing reference (for jsonl input)",
    )
    parser.add_argument(
        "--pred-field",
        help="Name of the field containing prediction (for jsonl input)",
    )
    parser.add_argument(
        "-a",
        "--align-types",
        help="Which file's tokenisation to use as reference for alignment. Valid choices are 'both', 'ref', 'pred'. Multiple choices are possible (comma separated)",
        required=True,
    )
    parser.add_argument(
        "--sent-wise-file",
        type=str,
        help="Path to a file where the sentence-wise accuracy scores get saved. For pickled output (list), the path must match /*.pkl/. Textual output is a comma-separated list",
    )
    # parser.add_argument('-c', '--cache', help='pickle file containing cached alignments', default=None)
    parser.add_argument(
        "--test-config",
        help="Path to the file containing the test configurations",
    )

    args = parser.parse_args(arguments)

    align_types = args.align_types.split(",")
    assert all(
        [x in ["both", "ref", "pred"] for x in align_types]
    ), 'Align types must belong to "both", "ref", "pred"'

    if args.input_type == "jsonl":
        if not args.ref_field or not args.pred_field:
            parser.error(
                "--ref-field and --pred-field are required when using jsonl format."
            )

    return args


def main(arguments: Optional[List[str]] = None) -> None:
    args = parse_and_check_arguments(arguments)

    if args.input_type == "jsonl":
        ref = read_jsonl_file(args.ref_file, args.ref_field)
        pred = read_jsonl_file(args.pred_file, args.pred_field)
    elif args.input_type == "text":
        ref = read_plain_text_file(args.ref_file)
        pred = read_plain_text_file(args.pred_file)

    align_types = args.align_types.split(",")

    metrics = get_metrics(ref, pred, align_types)

    # In case we computed sentence-wise scores: store them in file
    # Currently only accepts harmonized accuracy ("both")
    sent_wise_scores = metrics.pop("per_sent").get("both")
    pickle_output = re.match(r".*.pkl", args.sent_wise_file)
    if pickle_output:
        with open(args.sent_wise_file, "wb") as f:
            pickle.dump(sent_wise_scores, f)
    else:
        with open(args.sent_wise_file, "w", encoding="utf-8") as f:
            f.write(",".join([str(score) for score in sent_wise_scores]))

    output = {"pred-file": args.pred_file, "ref-file": args.ref_file}
    if args.test_config:
        output["test-config"] = args.test_config
    output.update(metrics)
    print(json.dumps(output))

    return


if __name__ == "__main__":
    main()
