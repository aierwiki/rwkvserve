"""
RWKVServe entrypoints module

Provides a vLLM-style offline batch inference interface.
"""

from rwkvserve.entrypoints.llm import LLM

__all__ = ["LLM"]
