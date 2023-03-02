from typing import List, Tuple


def markup_spans(
    text: str,
    spans: List[Tuple[int, int]],
    opening_tag: str = "<span>",
    closing_tag: str = "</span>",
) -> str:
    """Insert markup tags at index positions in a string"""
    if not text:
        return text
    marked_up_text = ""
    # Make sure spans are in correct order
    spans = sorted(spans)
    end_prev_span = 0
    # Build-up output str with mark-up tags
    for i, j in spans:
        if i < end_prev_span:
            raise ValueError("Overlapping spans are not supported.")
        if i > j:
            raise ValueError(
                "Spans (i,j), where i>j, are not supported. (Would garble output string.)"
            )
        marked_up_text += text[end_prev_span:i] + opening_tag + text[i:j] + closing_tag
        end_prev_span = j
    marked_up_text += text[end_prev_span:]
    return marked_up_text
