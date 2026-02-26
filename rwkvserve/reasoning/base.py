"""
Base reasoning parser interface (mirrors vLLM's ReasoningParser design).
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Tuple, Type, Union


class ReasoningParser(ABC):
    """
    Base class for reasoning parsers.

    A reasoning parser extracts thinking/reasoning content from model output,
    separating it from the final response content.
    """

    @abstractmethod
    def extract_reasoning_content(
        self, model_output: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content from a complete model output string.

        Args:
            model_output: Full model-generated text.

        Returns:
            (reasoning_content, content) tuple.
            reasoning_content: The thinking/reasoning portion (or None).
            content: The final answer portion (or None).
        """
        ...

    @abstractmethod
    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content from a partial (streaming) model output.

        Args:
            previous_text: Text generated so far before this step.
            current_text: Text generated so far including this step.
            delta_text: Newly generated text in this step.
            previous_token_ids: Token IDs before this step.
            current_token_ids: Token IDs including this step.
            delta_token_ids: Newly generated token IDs.

        Returns:
            (reasoning_delta, content_delta) tuple.
            reasoning_delta: New reasoning text in this step (or None).
            content_delta: New content text in this step (or None).
        """
        ...


class ReasoningParserManager:
    """Registry for reasoning parsers (mirrors vLLM's pattern)."""

    _parsers: Dict[str, Type[ReasoningParser]] = {}

    @classmethod
    def register(cls, names: list):
        """Decorator to register a reasoning parser under one or more names."""
        def decorator(parser_cls: Type[ReasoningParser]):
            for name in names:
                cls._parsers[name] = parser_cls
            return parser_cls
        return decorator

    @classmethod
    def get_parser(cls, name: str) -> ReasoningParser:
        if name not in cls._parsers:
            available = ", ".join(cls._parsers.keys()) or "none"
            raise ValueError(
                f"Unknown reasoning parser: {name!r}. Available: {available}"
            )
        return cls._parsers[name]()

    @classmethod
    def available_parsers(cls) -> list:
        return list(cls._parsers.keys())
