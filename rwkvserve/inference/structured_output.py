"""
Structured Output Module - JSON Schema Constrained Generation

Uses the lm-format-enforcer library to enforce JSON Schema constraints during
sampling by masking tokens that violate the schema, ensuring correct output format.

Supports:
- JSON Schema constraints (OpenAI-compatible response_format)
- JSON Object mode (forces valid JSON output)

Usage:
    from rwkvserve.inference.structured_output import JsonSchemaEnforcer

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    enforcer = JsonSchemaEnforcer(tokenizer, schema)

    logits = enforcer.mask_logits(logits)
    token = sample(logits)
    is_done = enforcer.update(token)
"""

import torch
from typing import Any, Callable, Dict, List, Optional, Union
import json

try:
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer, CharacterLevelParser
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data,
    )
    LM_FORMAT_ENFORCER_AVAILABLE = True
except ImportError:
    LM_FORMAT_ENFORCER_AVAILABLE = False


class TokenizerAdapter:
    """
    Tokenizer adapter that adds attributes required by lm-format-enforcer.

    lm-format-enforcer requires:
    - all_special_ids: list of special token IDs
    - vocab_size: vocabulary size
    - convert_ids_to_tokens: convert IDs to token strings

    RWKV tokenizer specifics:
    - Tokens 0-255 are typically byte-level tokens, many decode to invalid characters
    - These invalid tokens are added to all_special_ids to exclude them
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

        if hasattr(tokenizer, 'vocab_size'):
            vocab_size = tokenizer.vocab_size
        elif hasattr(tokenizer, 'get_vocab'):
            vocab_size = len(tokenizer.get_vocab())
        else:
            vocab_size = 65536
        self._vocab_size = vocab_size

        special_ids = set()

        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            special_ids.add(tokenizer.eos_token_id)
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            special_ids.add(tokenizer.bos_token_id)
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            special_ids.add(tokenizer.pad_token_id)
        if hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None:
            special_ids.add(tokenizer.unk_token_id)

        for tid in range(min(512, vocab_size)):
            try:
                decoded = tokenizer.decode([tid])
                if not decoded:
                    special_ids.add(tid)
                elif decoded == '\ufffd' or (len(decoded) == 1 and ord(decoded) < 32 and decoded not in '\n\r\t'):
                    special_ids.add(tid)
            except:
                special_ids.add(tid)

        self.all_special_ids = list(special_ids)

    def __getattr__(self, name):
        """Proxy other attributes to the original tokenizer."""
        return getattr(self._tokenizer, name)

    def convert_ids_to_tokens(self, ids):
        """Convert ID list to token string list."""
        if hasattr(self._tokenizer, 'convert_ids_to_tokens'):
            return self._tokenizer.convert_ids_to_tokens(ids)

        if isinstance(ids, int):
            return self._tokenizer.decode([ids])
        return [self._tokenizer.decode([i]) for i in ids]

    @property
    def vocab_size(self):
        return self._vocab_size

    def __len__(self):
        return self.vocab_size

    def get_vocab(self):
        """Get vocabulary (required by lm-format-enforcer)."""
        if hasattr(self._tokenizer, 'get_vocab'):
            return self._tokenizer.get_vocab()
        return {self._tokenizer.decode([i]): i for i in range(min(self.vocab_size, 1000))}


class JsonSchemaEnforcer:
    """
    JSON Schema constrained sampler.

    Uses lm-format-enforcer to force model output to conform to a JSON Schema
    by masking disallowed tokens during sampling.

    Args:
        tokenizer: HuggingFace-style tokenizer
        schema: JSON Schema dictionary

    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "tags": {
        ...             "type": "array",
        ...             "items": {"type": "string"},
        ...             "minItems": 5,
        ...             "maxItems": 5
        ...         }
        ...     },
        ...     "required": ["tags"]
        ... }
        >>> enforcer = JsonSchemaEnforcer(tokenizer, schema)
        >>> logits = enforcer.mask_logits(logits)
        >>> token = sample(logits)
        >>> is_done = enforcer.update(token)
    """

    def __init__(self, tokenizer, schema: Dict[str, Any]):
        if not LM_FORMAT_ENFORCER_AVAILABLE:
            raise ImportError(
                "lm-format-enforcer is required for JSON Schema enforcement. "
                "Install it with: pip install lm-format-enforcer"
            )

        self.tokenizer = tokenizer
        self.schema = schema

        self.adapted_tokenizer = TokenizerAdapter(tokenizer)
        self.tokenizer_data = build_token_enforcer_tokenizer_data(self.adapted_tokenizer)
        self.parser = JsonSchemaParser(schema)
        self.enforcer = TokenEnforcer(self.tokenizer_data, self.parser)

        self._generated_tokens: List[int] = []
        self._is_finished = False
        self._invalid_tokens: Optional[set] = None

    def _get_invalid_tokens(self) -> set:
        """Get set of invalid tokens (lazy-loaded)."""
        if self._invalid_tokens is None:
            self._invalid_tokens = set()
            vocab_size = self.adapted_tokenizer.vocab_size
            for tid in range(vocab_size):
                try:
                    decoded = self.tokenizer.decode([tid])
                    if decoded == '\ufffd':
                        self._invalid_tokens.add(tid)
                except:
                    self._invalid_tokens.add(tid)
        return self._invalid_tokens

    def get_allowed_tokens(self) -> List[int]:
        """Get currently allowed token IDs (with invalid tokens filtered out)."""
        token_list = self.enforcer.get_allowed_tokens(self._generated_tokens)
        allowed = token_list.allowed_tokens if token_list.allowed_tokens else []
        invalid = self._get_invalid_tokens()
        return [tid for tid in allowed if tid not in invalid]

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Mask disallowed token logits to -inf.

        Args:
            logits: [vocab_size] or [batch_size, vocab_size]

        Returns:
            Masked logits tensor (same shape as input)
        """
        if self._is_finished:
            return logits

        allowed_tokens = self.get_allowed_tokens()

        if not allowed_tokens:
            self._is_finished = True
            return logits

        if logits.dim() == 1:
            mask = torch.full_like(logits, float('-inf'))
            mask[allowed_tokens] = 0
            return logits + mask
        elif logits.dim() == 2:
            mask = torch.full_like(logits, float('-inf'))
            mask[:, allowed_tokens] = 0
            return logits + mask
        else:
            raise ValueError(f"Unsupported logits shape: {logits.shape}")

    def update(self, token_id: int) -> bool:
        """
        Update state machine after sampling a token.

        Args:
            token_id: Sampled token ID

        Returns:
            Whether generation is complete
        """
        self._generated_tokens.append(token_id)

        try:
            allowed = self.get_allowed_tokens()
            if not allowed:
                self._is_finished = True
            if hasattr(self.enforcer, 'eos_token_id') and token_id == self.enforcer.eos_token_id:
                self._is_finished = True
        except Exception as e:
            print(f"Warning: JsonSchemaEnforcer update error: {e}")
            self._is_finished = True

        return self._is_finished

    def is_finished(self) -> bool:
        """Check whether generation is complete."""
        return self._is_finished

    def reset(self):
        """Reset state machine for a new generation."""
        self._generated_tokens = []
        self._is_finished = False


class JsonObjectEnforcer:
    """
    JSON Object constrained sampler (simplified version).

    Forces the model to output valid JSON objects without restricting structure.
    More lenient than JsonSchemaEnforcer - only ensures output is valid JSON.
    """

    def __init__(self, tokenizer):
        if not LM_FORMAT_ENFORCER_AVAILABLE:
            raise ImportError(
                "lm-format-enforcer is required for JSON enforcement. "
                "Install it with: pip install lm-format-enforcer"
            )

        self.tokenizer = tokenizer
        self.adapted_tokenizer = TokenizerAdapter(tokenizer)
        self.tokenizer_data = build_token_enforcer_tokenizer_data(self.adapted_tokenizer)

        self.schema = {"type": "object"}
        self.parser = JsonSchemaParser(self.schema)
        self.enforcer = TokenEnforcer(self.tokenizer_data, self.parser)

        self._generated_tokens: List[int] = []
        self._is_finished = False
        self._invalid_tokens: Optional[set] = None

    def _get_invalid_tokens(self) -> set:
        """Get set of invalid tokens (lazy-loaded)."""
        if self._invalid_tokens is None:
            self._invalid_tokens = set()
            vocab_size = self.adapted_tokenizer.vocab_size
            for tid in range(vocab_size):
                try:
                    decoded = self.tokenizer.decode([tid])
                    if decoded == '\ufffd':
                        self._invalid_tokens.add(tid)
                except:
                    self._invalid_tokens.add(tid)
        return self._invalid_tokens

    def get_allowed_tokens(self) -> List[int]:
        """Get currently allowed token IDs (with invalid tokens filtered out)."""
        token_list = self.enforcer.get_allowed_tokens(self._generated_tokens)
        allowed = token_list.allowed_tokens if token_list.allowed_tokens else []
        invalid = self._get_invalid_tokens()
        return [tid for tid in allowed if tid not in invalid]

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask disallowed token logits to -inf."""
        if self._is_finished:
            return logits

        allowed_tokens = self.get_allowed_tokens()

        if not allowed_tokens:
            self._is_finished = True
            return logits

        if logits.dim() == 1:
            mask = torch.full_like(logits, float('-inf'))
            mask[allowed_tokens] = 0
            return logits + mask
        elif logits.dim() == 2:
            mask = torch.full_like(logits, float('-inf'))
            mask[:, allowed_tokens] = 0
            return logits + mask
        else:
            raise ValueError(f"Unsupported logits shape: {logits.shape}")

    def update(self, token_id: int) -> bool:
        """Update state machine after sampling a token."""
        self._generated_tokens.append(token_id)

        try:
            allowed = self.get_allowed_tokens()
            if not allowed:
                self._is_finished = True
            if hasattr(self.enforcer, 'eos_token_id') and token_id == self.enforcer.eos_token_id:
                self._is_finished = True
        except Exception as e:
            print(f"Warning: JsonObjectEnforcer update error: {e}")
            self._is_finished = True

        return self._is_finished

    def is_finished(self) -> bool:
        """Check whether generation is complete."""
        return self._is_finished

    def reset(self):
        """Reset state machine for a new generation."""
        self._generated_tokens = []
        self._is_finished = False


def create_enforcer(
    tokenizer,
    response_format: Optional[Dict[str, Any]] = None,
) -> Optional[Union[JsonSchemaEnforcer, JsonObjectEnforcer]]:
    """
    Create an enforcer based on response_format.

    Args:
        tokenizer: Tokenizer object
        response_format: OpenAI-style response_format configuration
            - {"type": "text"}: no constraints (returns None)
            - {"type": "json_object"}: uses JsonObjectEnforcer
            - {"type": "json_schema", "json_schema": {...}}: uses JsonSchemaEnforcer

    Returns:
        Enforcer object, or None if no constraints needed

    Example:
        >>> response_format = {
        ...     "type": "json_schema",
        ...     "json_schema": {
        ...         "name": "tags",
        ...         "schema": {"type": "object", "properties": {"tags": {"type": "array"}}}
        ...     }
        ... }
        >>> enforcer = create_enforcer(tokenizer, response_format)
    """
    if response_format is None:
        return None

    format_type = response_format.get("type", "text")

    if format_type == "text":
        return None

    elif format_type == "json_object":
        return JsonObjectEnforcer(tokenizer)

    elif format_type == "json_schema":
        json_schema_config = response_format.get("json_schema", {})
        schema = json_schema_config.get("schema") or json_schema_config.get("schema_", {})
        if not schema:
            raise ValueError("json_schema.schema is required when type is 'json_schema'")
        return JsonSchemaEnforcer(tokenizer, schema)

    else:
        raise ValueError(f"Unsupported response_format type: {format_type}")


def is_structured_output_available() -> bool:
    """Check whether structured output functionality is available."""
    return LM_FORMAT_ENFORCER_AVAILABLE
