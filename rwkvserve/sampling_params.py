"""
Sampling parameter definitions - vLLM-style API

Provides a vLLM-like SamplingParams interface for controlling text generation behavior.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union


@dataclass
class SamplingParams:
    """
    Sampling parameters for controlling text generation behavior.

    Simplified version of vLLM's SamplingParams, tailored for RWKV models.

    Args:
        max_tokens: Maximum number of tokens to generate (default: 256).
        temperature: Sampling temperature. Higher values increase randomness,
            lower values increase determinism. 0 means greedy sampling.
        top_p: Nucleus sampling parameter. Retains tokens with cumulative
            probability >= top_p.
        top_k: Top-k sampling parameter. 0 means no restriction.
        stop: List of stop strings.
        stop_token_ids: List of stop token IDs.
        skip_special_tokens: Whether to skip special tokens during decoding.
        repetition_penalty: Repetition penalty factor.
        logit_bias: Mapping of token ID to bias value. -100 means the token
            is forbidden from being generated.

    Example:
        >>> params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)
        >>> params = SamplingParams(logit_bias={"65530": -100})
        >>> llm.generate(prompts, params)
    """
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    skip_special_tokens: bool = True
    repetition_penalty: float = 1.0
    logit_bias: Optional[dict] = None

    def __post_init__(self):
        """Validate parameters."""
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0 <= self.top_p <= 1:
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")

        if self.stop is None:
            self.stop = []
        if self.stop_token_ids is None:
            self.stop_token_ids = []

    @classmethod
    def from_dict(cls, d: dict) -> "SamplingParams":
        """Create SamplingParams from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop": self.stop,
            "stop_token_ids": self.stop_token_ids,
            "skip_special_tokens": self.skip_special_tokens,
            "repetition_penalty": self.repetition_penalty,
            "logit_bias": self.logit_bias,
        }
