"""
Tokenizers for RWKV models.
"""

import os
from typing import Optional
from rwkvserve.data.tokenizers.rwkv_tokenizer import TRIE_TOKENIZER
from rwkvserve.data.tokenizers.auto_tokenizer import (
    AutoTokenizer,
    RWKVTokenizer,
    TokenizerConfig,
)

__all__ = [
    "TRIE_TOKENIZER",
    "AutoTokenizer",
    "RWKVTokenizer",
    "TokenizerConfig",
    "get_tokenizer",
    "get_vocab_path",
]


def get_vocab_path(vocab_name: str = "rwkv_vocab_v20230424.txt") -> str:
    """
    Get the path to the RWKV vocabulary file.

    Args:
        vocab_name: Vocabulary filename, defaults to "rwkv_vocab_v20230424.txt"

    Returns:
        Absolute path to the vocabulary file.

    Raises:
        FileNotFoundError: If the vocabulary file does not exist.
    """
    tokenizers_dir = os.path.dirname(__file__)
    vocab_path = os.path.join(tokenizers_dir, "vocab", vocab_name)

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"Vocabulary file not found: {vocab_path}\n"
            f"Expected location: {os.path.join(tokenizers_dir, 'vocab')}"
        )

    return vocab_path


def get_tokenizer(
    vocab_path: Optional[str] = None,
    vocab_name: str = "rwkv_vocab_v20230424.txt",
    eos_token: Optional[str] = None,
) -> TRIE_TOKENIZER:
    """
    Get an RWKV tokenizer instance.

    Convenience function to load the RWKV TRIE_TOKENIZER.
    If vocab_path is not specified, uses the built-in default vocabulary.

    Args:
        vocab_path: Full path to the vocabulary file. If None, uses the default.
        vocab_name: Vocabulary filename when vocab_path is None.
        eos_token: EOS (End-of-Sequence) token, defaults to '\\n\\n'.

    Returns:
        TRIE_TOKENIZER instance.
    """
    if vocab_path is None:
        vocab_path = get_vocab_path(vocab_name)

    return TRIE_TOKENIZER(vocab_path, eos_token=eos_token)
