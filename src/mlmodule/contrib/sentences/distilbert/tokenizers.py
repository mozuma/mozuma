import os

from tokenizers import Tokenizer


def get_distil_bert_tokenizer() -> Tokenizer:
    """Loads a tokenizers.Tokenizer from the local tokenizer.json file"""
    return Tokenizer.from_file(
        os.path.join(os.path.dirname(__file__), "tokenizer.json")
    )
