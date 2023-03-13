from transformers import AutoTokenizer
from transnormer.preprocess import translit


def test_exchange_transliterator():
    # get a tokenizer
    tokenizer_orig = AutoTokenizer.from_pretrained(
        "dbmdz/bert-base-historic-multilingual-cased"
    )
    # Exchanging the transliterator with Transliterator1
    transliterator1 = translit.Transliterator1()
    tokenizer_updated = translit.exchange_transliterator(
        tokenizer_orig, transliterator=transliterator1
    )
    assert tokenizer_orig != tokenizer_updated

    # When transliterator=None, the original tokenizer is returned unchanged
    tokenizer_updated = translit.exchange_transliterator(
        tokenizer_orig, transliterator=None
    )
    assert tokenizer_orig is tokenizer_updated


def test_transliterator1():
    # get a tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "dbmdz/bert-base-historic-multilingual-cased"
    )
    # sentence to tokenize, contains Schaft-s and combining small letter e
    source_str = "wie erhebt er ſich durch Woͤrter!"

    # tokenize it and join again without custom transliterator
    toktext = tokenizer(source_str, add_special_tokens=False)
    output = tokenizer.decode(toktext.input_ids)
    target_str = "wie erhebt er [UNK] durch [UNK]!"
    assert target_str == output

    # Now replace the default transliterator with Transliterator1
    transliterator1 = translit.Transliterator1()
    tokenizer = translit.exchange_transliterator(tokenizer, transliterator1)
    # tokenize and join with Transliterator1
    toktext = tokenizer(source_str, add_special_tokens=False)
    output = tokenizer.decode(toktext.input_ids)
    target_str = "wie erhebt er sich durch Woerter!"
    assert target_str == output
