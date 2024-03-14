from transnormer.evaluation.dataset_stats import type_alignment_stats

SENTS_ORIG = [
    "Womit aber der von Fliſco nicht allerdings will einſtimmen.",
    "Und das ſeufftzen/ das wir treiben/ hilfft der leichte Wind verſtaͤuben.",
    # "Die Sternen tilgt der Tag/ Kometen werden graus.",
    # "Das Haupt und der übrige Leib biß an den Nabel/ waren Menſchlich gebildet:",
]
SENTS_NORM = [
    "Womit aber der von Flisco nicht allerdings will einstimmen.",
    "Und das seufzen/ das wir treiben/ hilft der leichte Wind verstäuben.",
    # "Die Sternen tilgt der Tag/ Kometen werden graus.",
    # "Das Haupt und der übrige Leib bis an den Nabel/ waren menschlich gebildet:"
]
SENTS_PRED = [
    "Womit aber der von Flisco nicht allerdings will einstimmen.",
    "Und das seufzen/ das wir treiben/ hilft der leichte Wind verstäuben."
    # "Die Sternen tilgt der Tag/ Kometen werden graus.",
    # "Das Haupt und der übrige Leib bis an den Nabel/ waren Menschlich gebildet:"
]


def test_type_alignment_stats() -> None:
    filtered_stats = type_alignment_stats(SENTS_ORIG, SENTS_NORM)
    target = {
        "Womit": {
            "Womit": 1,
        },
        "aber": {
            "aber": 1,
        },
        "der": {
            "der": 2,
        },
        "von": {
            "von": 1,
        },
        "Fliſco": {
            "Flisco": 1,
        },
        "nicht": {
            "nicht": 1,
        },
        "allerdings": {
            "allerdings": 1,
        },
        "will": {
            "will": 1,
        },
        "einſtimmen": {
            "einstimmen": 1,
        },
        "Und": {
            "Und": 1,
        },
        "das": {
            "das": 2,
        },
        "ſeufftzen": {
            "seufzen": 1,
        },
        "/": {
            "/": 2,
        },
        "wir": {
            "wir": 1,
        },
        "treiben": {
            "treiben": 1,
        },
        "hilfft": {
            "hilft": 1,
        },
        "leichte": {
            "leichte": 1,
        },
        "Wind": {
            "Wind": 1,
        },
        "verſtaͤuben": {
            "verstäuben": 1,
        },
        ".": {
            ".": 2,
        },
    }
    assert filtered_stats == target
