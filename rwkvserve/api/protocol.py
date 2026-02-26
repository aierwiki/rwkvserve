"""
OpenAI API Protocol Models

Defines request and response data models compatible with the OpenAI API.
"""

import time
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# ========== Base Models ==========

class OpenAIBaseModel(BaseModel):
    """Base model compatible with OpenAI API, allows extra fields for extra_body support."""
    model_config = ConfigDict(extra="allow")

class ErrorResponse(BaseModel):
    """Error response."""
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class UsageInfo(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ModelCard(BaseModel):
    """Model information card."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "rwkvserve"


class ModelList(BaseModel):
    """Model list."""
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


# ========== Structured Output (JSON Schema) ==========

class JsonSchema(BaseModel):
    """JSON Schema definition."""
    name: str
    description: Optional[str] = None
    schema_: Dict[str, Any] = Field(alias="schema")
    strict: Optional[bool] = None

    model_config = {"populate_by_name": True}


class ResponseFormat(BaseModel):
    """
    Response format configuration (OpenAI compatible).

    Supported modes:
    - type="text": Plain text output (default)
    - type="json_object": Force valid JSON output
    - type="json_schema": Force output conforming to a specified JSON Schema
    """
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: Optional[JsonSchema] = None


# ========== Chat Completion API ==========

class ChatMessage(BaseModel):
    """Chat message."""
    role: Literal["system", "user", "assistant"] = "user"
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(OpenAIBaseModel):
    """Chat completion request."""
    model: str
    messages: List[ChatMessage]
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 100
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 100
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    extra_body: Optional[Dict[str, Any]] = None
    response_format: Optional[ResponseFormat] = None


class ChatCompletionResponseChoice(BaseModel):
    """Chat completion response choice."""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    """Streaming response delta message."""
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None


class ChatCompletionStreamResponseChoice(BaseModel):
    """Streaming chat completion response choice."""
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """Streaming chat completion response."""
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamResponseChoice]


# ========== Completion API ==========

class CompletionRequest(OpenAIBaseModel):
    """Text completion request."""
    model: str
    prompt: Union[str, List[str]]
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 100
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 100
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    echo: bool = False
    logprobs: Optional[int] = None
    suffix: Optional[str] = None
    best_of: Optional[int] = None
    extra_body: Optional[Dict[str, Any]] = None


class CompletionResponseChoice(BaseModel):
    """Text completion response choice."""
    index: int
    text: str
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


class CompletionResponse(BaseModel):
    """Text completion response."""
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionStreamResponseChoice(BaseModel):
    """Streaming text completion response choice."""
    index: int
    text: str
    finish_reason: Optional[str] = None


class CompletionStreamResponse(BaseModel):
    """Streaming text completion response."""
    id: str
    object: str = "text_completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionStreamResponseChoice]

