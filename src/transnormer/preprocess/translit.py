from copy import deepcopy
from typing import Optional

from tokenizers.normalizers import Normalizer
from tokenizers import NormalizedString

from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class BaseTransliterator:
    def normalize(self, normalized):
        pass


class Transliterator1(BaseTransliterator):
    def normalize(self, normalized: NormalizedString):
        # Unicode decomposition of combining characters
        normalized.nfd()
        # Long s -> normal s
        normalized.replace("ſ", "s")
        # Round r -> normal r
        normalized.replace("ꝛ", "r")
        # Drop combining tilde
        normalized.replace(chr(0x0303), "")
        # "Combining Latin Small Letter E" ->  normal e
        normalized.replace(chr(0x0364), "e")
        # More conversions of Umlaut-like chars
        normalized.replace("æ", "ae")
        normalized.replace("ů", "ü")
        normalized.replace("Ů", "Ü")
        # Unicode composition (put decomposed chars back together)
        normalized.nfc()


def exchange_transliterator(
    tokenizer: PreTrainedTokenizerBase,
    transliterator: Optional[BaseTransliterator] = None,
) -> PreTrainedTokenizerBase:
    """
    Exchange the normalizer component of a huggingface tokenizer with a
    Transliterator defined in `transnormer.preprocess.translit`

    If `transliterator=None`, the original tokenizer is returned
    """
    # Passing any custom transliterator
    if transliterator:
        tokenizer_with_new_transliterator = deepcopy(tokenizer)
        tokenizer_with_new_transliterator.backend_tokenizer.normalizer = (
            Normalizer.custom(transliterator)
        )
        return tokenizer_with_new_transliterator
    # Passing None does not change the transliterator
    else:
        return tokenizer
