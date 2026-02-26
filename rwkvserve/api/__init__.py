"""
RWKVServe OpenAI-Compatible API Server

Provides OpenAI-compatible API endpoints for RWKV model inference.

Features:
- Continuous Batching: Multiple requests share GPU compute
- n>1 + streaming: Generate multiple candidate replies simultaneously
- State Cache: Accelerate similar requests
- Multi-model deployment
"""

from rwkvserve.api.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
    ModelCard,
    ModelList,
)
from rwkvserve.api.model_manager import (
    ModelManager,
    ModelConfig,
    LoadedModel,
)
from rwkvserve.api.api_server import create_app
from rwkvserve.api.async_serving_chat import AsyncChatCompletionService
from rwkvserve.api.async_serving_completion import AsyncCompletionService

__all__ = [
    # Protocol
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "CompletionRequest",
    "CompletionResponse",
    "ErrorResponse",
    "ModelCard",
    "ModelList",
    # Model Manager
    "ModelManager",
    "ModelConfig",
    "LoadedModel",
    # API Server
    "create_app",
    # Services
    "AsyncChatCompletionService",
    "AsyncCompletionService",
]

