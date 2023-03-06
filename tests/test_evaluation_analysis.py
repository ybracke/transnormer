from transformers import AutoTokenizer
from transnormer.evaluation import analysis


def test_get_spans_of_unknown_tokens():
    text = "This ſentence contains unknowable ſymbols"
    tokenizer_checkpoint = "dbmdz/bert-base-historic-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    target_spans = [(5, 13), (34, 41)]
    target_unk_toks = ["ſentence", "ſymbols"]
    actual_spans = analysis.get_spans_of_unknown_tokens(text, tokenizer)
    actual_unk_toks = [text[i:j] for (i, j) in actual_spans]
    assert target_spans == actual_spans
    assert target_unk_toks == actual_unk_toks
