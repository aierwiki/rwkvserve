"""
Output type definitions - vLLM-style API

Provides a vLLM-like RequestOutput interface for returning generation results.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CompletionOutput:
    """
    Output for a single completion.

    Attributes:
        index: Output index (a single prompt may produce multiple completions).
        text: Generated text.
        token_ids: List of generated token IDs.
        cumulative_logprob: Cumulative log probability (optional).
        finish_reason: Reason for finishing ("stop", "length", or None).
    """
    index: int
    text: str
    token_ids: List[int] = field(default_factory=list)
    cumulative_logprob: Optional[float] = None
    finish_reason: Optional[str] = None
    
    def __repr__(self) -> str:
        return (
            f"CompletionOutput(index={self.index}, "
            f"text={self.text[:50]!r}{'...' if len(self.text) > 50 else ''}, "
            f"finish_reason={self.finish_reason!r})"
        )


@dataclass
class RequestOutput:
    """
    Complete output for a single request.

    Attributes:
        request_id: Request ID.
        prompt: Original prompt text.
        prompt_token_ids: List of prompt token IDs.
        outputs: List of generated completions.
        finished: Whether generation is complete.

    Example:
        >>> output = llm.generate("Hello")[0]
        >>> print(output.prompt)
        "Hello"
        >>> print(output.outputs[0].text)
        " world! How are you?"
    """
    request_id: str
    prompt: str
    prompt_token_ids: List[int] = field(default_factory=list)
    outputs: List[CompletionOutput] = field(default_factory=list)
    finished: bool = True
    
    def __repr__(self) -> str:
        return (
            f"RequestOutput(request_id={self.request_id!r}, "
            f"prompt={self.prompt[:30]!r}{'...' if len(self.prompt) > 30 else ''}, "
            f"outputs={len(self.outputs)} completions)"
        )
