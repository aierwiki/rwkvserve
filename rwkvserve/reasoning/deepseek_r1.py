"""
DeepSeek-R1 style reasoning parser.

Extracts content between <think>...</think> tags as reasoning_content,
and everything after </think> as content.

Compatible with any model that uses <think>...</think> for chain-of-thought,
including RWKV models trained with thinking mode.
"""

import re
from typing import Optional, Sequence, Tuple

from rwkvserve.reasoning.base import ReasoningParser, ReasoningParserManager

THINK_START = "<think>"
THINK_END = "</think>"


@ReasoningParserManager.register(["deepseek_r1", "think"])
class DeepSeekR1ReasoningParser(ReasoningParser):
    """
    Parser for <think>...</think> reasoning format.

    Input:  "<think>step-by-step reasoning</think>final answer"
    Output: reasoning_content="step-by-step reasoning", content="final answer"

    Also handles cases where:
    - No <think> tag present (all goes to content)
    - <think> present but no </think> yet (all goes to reasoning, streaming)
    - Multiple <think> blocks (only first is extracted)
    """

    def extract_reasoning_content(
        self, model_output: str
    ) -> Tuple[Optional[str], Optional[str]]:
        if not model_output:
            return None, None

        text = model_output.strip()

        # Try to find <think>...</think> pattern
        match = re.search(
            r"<think>(.*?)</think>(.*)",
            text,
            re.DOTALL,
        )

        if match:
            reasoning = match.group(1).strip()
            content = match.group(2).strip()
            return reasoning or None, content or None

        # If text starts with <think> but no closing tag, all is reasoning
        if text.startswith(THINK_START):
            reasoning = text[len(THINK_START):].strip()
            return reasoning or None, None

        # No thinking tags at all
        return None, text or None

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Tuple[Optional[str], Optional[str]]:
        think_end_pos_prev = previous_text.find(THINK_END)
        think_end_pos_curr = current_text.find(THINK_END)

        # Case 1: </think> was already found before this delta — all new text is content
        if think_end_pos_prev >= 0:
            return None, delta_text

        # Case 2: </think> appears for the first time in this delta
        if think_end_pos_curr >= 0:
            end_tag_start = think_end_pos_curr
            after_tag = current_text[end_tag_start + len(THINK_END):]
            # The content delta is whatever comes after </think> in current_text
            # minus what was already emitted
            content_in_delta = after_tag  # this is the new content portion
            return None, content_in_delta.strip() if content_in_delta.strip() else None

        # Case 3: still inside <think> block
        # Skip the <think> tag itself if it's in the delta
        if THINK_START in delta_text:
            reasoning_part = delta_text.replace(THINK_START, "", 1)
            return reasoning_part if reasoning_part else None, None

        # Partial </think> tag might be forming — need to buffer carefully
        # Check if current_text ends with a partial match of "</think>"
        for partial_len in range(1, len(THINK_END)):
            if current_text.endswith(THINK_END[:partial_len]):
                safe_delta = delta_text[: len(delta_text) - partial_len]
                return safe_delta if safe_delta else None, None

        return delta_text if delta_text else None, None
