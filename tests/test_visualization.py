import pytest

from transnormer.visualization.formatting import markup_spans


def test_markup_spans_normal():
    # Test function with a "normal" input string and two spans
    spans = [(0, 4), (8, 10)]
    text = "This is it."
    target = "<span>This</span> is <span>it</span>."
    assert target == markup_spans(text, spans)


def test_markup_spans_full_string():
    # Test function with a span stretching the entire string
    spans = [(0, 11)]
    text = "This is it."
    target = "<span>This is it.</span>"
    assert target == markup_spans(text, spans)


def test_markup_spans_empty_string_empty_spans():
    # Test function with an empty string and no markup_spans
    spans = []
    text = ""
    target = ""
    assert target == markup_spans(text, spans)


def test_markup_spans_empty_string_nonempty_spans():
    spans = [(0, 4), (8, 10)]
    text = ""
    target = ""
    assert target == markup_spans(text, spans)


def test_markup_spans_overlapping_spans():
    # This should throw an error
    spans = [(0, 4), (3, 10)]
    text = "This is it."
    with pytest.raises(ValueError, match=r"Overlapping spans are not supported."):
        _ = markup_spans(text, spans)


def test_markup_spans_weird_spans():
    # This should throw an error
    spans = [(8, 4)]
    text = "This is it."
    with pytest.raises(
        ValueError, match=r"Spans \(i,j\), where i>j, are not supported.*"
    ):
        _ = markup_spans(text, spans)
