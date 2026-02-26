"""
OpenAI-Compatible API Server - Multi-Model Deployment

Features:
- Load and serve multiple models simultaneously
- Automatic routing based on the model field in requests
- Per-model device assignment
- API Key authentication
- Request logging
- State Cache with multi-level caching and eviction
- Chunked Prefill for memory control
"""

import argparse
import hashlib
import json
import os
import secrets
import torch
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Awaitable, List, Optional, Union, Any, Dict

import uvicorn
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.datastructures import Headers, URL
from starlette.types import ASGIApp, Receive, Scope, Send

from rwkvserve.api.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
    ModelCard,
    ModelList,
)
from rwkvserve.api.model_manager import ModelManager, ModelConfig, LoadedModel


model_manager: ModelManager = None

request_logger: Optional["RequestLogger"] = None

global_cache_config: Optional[Dict[str, Any]] = None


class RequestLogger:
    """Logs all request inputs, inference parameters, and outputs to disk."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.chat_dir = self.log_dir / "chat_completions"
        self.completion_dir = self.log_dir / "completions"
        self.chat_dir.mkdir(exist_ok=True)
        self.completion_dir.mkdir(exist_ok=True)
        
        self._request_count = 0
        print(f"Request logging enabled, directory: {self.log_dir}")
    
    def _generate_filename(self, prefix: str, model: str) -> str:
        self._request_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_model = model.replace("/", "_").replace(":", "_")
        return f"{prefix}_{timestamp}_{safe_model}_{self._request_count:06d}.json"
    
    def log_chat_completion(
        self,
        request: Dict[str, Any],
        response: Dict[str, Any],
        model: str,
        duration_ms: float = 0,
    ) -> str:
        """Log a chat completion request and response."""
        filename = self._generate_filename("chat", model)
        log_path = self.chat_dir / filename
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "chat_completion",
            "model": model,
            "duration_ms": duration_ms,
            "request": request,
            "response": response,
        }
        
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        return str(log_path)
    
    def log_chat_completion_stream(
        self,
        request: Dict[str, Any],
        chunks: List[str],
        model: str,
        duration_ms: float = 0,
    ) -> str:
        """Log a streaming chat completion request."""
        filename = self._generate_filename("chat_stream", model)
        log_path = self.chat_dir / filename
        
        content_parts = []
        for chunk in chunks:
            if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                try:
                    data = json.loads(chunk[6:])
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            content_parts.append(delta["content"])
                except json.JSONDecodeError:
                    pass
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "chat_completion_stream",
            "model": model,
            "duration_ms": duration_ms,
            "request": request,
            "response": {
                "assembled_content": "".join(content_parts),
                "chunk_count": len(chunks),
                "raw_chunks": chunks,
            },
        }
        
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        return str(log_path)
    
    def log_completion_stream(
        self,
        request: Dict[str, Any],
        chunks: List[str],
        model: str,
        duration_ms: float = 0,
    ) -> str:
        """Log a streaming text completion request."""
        filename = self._generate_filename("completion_stream", model)
        log_path = self.completion_dir / filename
        
        content_parts = []
        for chunk in chunks:
            if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                try:
                    data = json.loads(chunk[6:])
                    if "choices" in data and len(data["choices"]) > 0:
                        text = data["choices"][0].get("text", "")
                        if text:
                            content_parts.append(text)
                except json.JSONDecodeError:
                    pass
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "completion_stream",
            "model": model,
            "duration_ms": duration_ms,
            "request": request,
            "response": {
                "assembled_content": "".join(content_parts),
                "chunk_count": len(chunks),
                "raw_chunks": chunks,
            },
        }
        
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        return str(log_path)
    
    def log_completion(
        self,
        request: Dict[str, Any],
        response: Dict[str, Any],
        model: str,
        duration_ms: float = 0,
    ) -> str:
        """Log a text completion request."""
        filename = self._generate_filename("completion", model)
        log_path = self.completion_dir / filename
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "completion",
            "model": model,
            "duration_ms": duration_ms,
            "request": request,
            "response": response,
        }
        
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        return str(log_path)


class APIKeyAuthMiddleware:
    """
    API Key authentication middleware.

    Verifies the Authorization Bearer token against configured API keys.
    Skips authentication for OPTIONS requests (CORS preflight) and
    non-/v1 paths (e.g. /health, /docs).
    """
    
    def __init__(self, app: ASGIApp, api_keys: List[str]) -> None:
        self.app = app
        self.api_key_hashes = [
            hashlib.sha256(key.encode("utf-8")).digest() 
            for key in api_keys
        ]
    
    def verify_token(self, headers: Headers) -> bool:
        """Verify the token in the Authorization header."""
        auth_header = headers.get("Authorization")
        if not auth_header:
            return False
        
        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() != "bearer":
            return False
        
        token_hash = hashlib.sha256(token.encode("utf-8")).digest()
        for key_hash in self.api_key_hashes:
            if secrets.compare_digest(token_hash, key_hash):
                return True
        
        return False
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        
        if scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return
        
        root_path = scope.get("root_path", "")
        url_path = URL(scope=scope).path.removeprefix(root_path)
        
        if url_path.startswith("/v1"):
            headers = Headers(scope=scope)
            if not self.verify_token(headers):
                response = JSONResponse(
                    content={"error": "Unauthorized", "message": "Invalid or missing API key"},
                    status_code=401
                )
                await response(scope, receive, send)
                return
        
        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    if model_manager:
        for model_name, loaded_model in model_manager.models.items():
            if loaded_model.is_data_parallel:
                print(f"Starting data-parallel engine: {model_name}")
                if hasattr(loaded_model.chat_service, "ensure_started"):
                    await loaded_model.chat_service.ensure_started()
            elif hasattr(loaded_model.chat_service, "start"):
                print(f"Starting background engine: {model_name}")
                await loaded_model.chat_service.start()
    
    yield
    
    print("Shutting down API server...")
    
    if model_manager:
        for model_name, loaded_model in model_manager.models.items():
            if loaded_model.is_data_parallel:
                print(f"Stopping data-parallel engine: {model_name}")
                if hasattr(loaded_model.chat_service, "shutdown"):
                    await loaded_model.chat_service.shutdown()


def create_app(
    models: Optional[List[ModelConfig]] = None,
    api_keys: Optional[List[str]] = None,
    log_requests: Optional[str] = None,
    cache_config: Optional[Dict[str, Any]] = None,
    model_path: Optional[str] = None,
    device: str = "cuda",
    model_name: str = "rwkv-7",
    dtype: Optional[str] = None,
    reasoning_parser=None,
) -> FastAPI:
    """
    Create the FastAPI application (supports multi-model deployment).

    Args:
        models: List of model configurations.
        api_keys: Optional list of API keys for authentication.
        log_requests: Optional directory to save request/response logs.
        cache_config: Optional State Cache configuration.
        model_path: Model path (legacy single-model interface).
        device: Device (legacy interface).
        model_name: Model name (legacy interface).
        dtype: Data type (legacy interface).

    Returns:
        FastAPI application instance.
    """
    global model_manager, request_logger, global_cache_config
    
    if log_requests:
        request_logger = RequestLogger(log_requests)
    else:
        request_logger = None
    
    global_cache_config = cache_config or {
        "max_cache_memory_gb": 4.0,
        "cache_level": "prefix",
        "cache_first_message": True,
    }
    
    model_manager = ModelManager(cache_config=global_cache_config, reasoning_parser=reasoning_parser)
    
    if models is None and model_path is not None:
        models = [ModelConfig(
            name=model_name,
            path=model_path,
            device=device,
            dtype=dtype,
        )]
    
    if models is None or len(models) == 0:
        raise ValueError("At least one model configuration is required")
    
    app = FastAPI(
        title="RWKVServe OpenAI-Compatible API",
        description="OpenAI-compatible API for RWKV models (Multi-Model Support)",
        version="2.1.0",
        lifespan=lifespan,
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    env_api_key = os.environ.get("RWKV_API_KEY")
    all_api_keys = list(api_keys or [])
    if env_api_key:
        all_api_keys.append(env_api_key)
    
    if all_api_keys:
        app.add_middleware(APIKeyAuthMiddleware, api_keys=all_api_keys)
        print(f"API Key authentication enabled ({len(all_api_keys)} keys)")
    
    print("=" * 60)
    print("Initializing RWKV models...")
    print("=" * 60)
    print(f"Models to load: {len(models)}")
    
    model_manager.load_models(models)
    
    print("=" * 60)
    print(f"All models loaded ({len(model_manager)} total)")
    print(f"Available models: {', '.join(model_manager.model_names)}")
    print(f"Default model: {model_manager.default_model}")
    print("=" * 60)
    
    @app.get("/")
    async def root():
        features = ["multi-model-support"]
        
        return {
            "message": "RWKVServe OpenAI-Compatible API Server",
            "version": "2.2.0",
            "features": features,
            "models": model_manager.model_names,
            "default_model": model_manager.default_model,
            "endpoints": {
                "models": "/v1/models",
                "chat": "/v1/chat/completions",
                "completions": "/v1/completions",
            }
        }
    
    @app.get("/health")
    async def health():
        health_info = {
            "status": "ok",
            "models_loaded": len(model_manager),
            "model_names": model_manager.model_names,
        }
        
        cache_stats = model_manager.get_cache_stats()
        if cache_stats:
            health_info["state_cache"] = cache_stats
        
        dp_models = []
        for name, loaded_model in model_manager.models.items():
            if loaded_model.is_data_parallel:
                dp_models.append(name)
        if dp_models:
            health_info["data_parallel_models"] = dp_models
        
        return health_info
    
    @app.get("/dp/stats")
    async def data_parallel_stats():
        """Return statistics for all data-parallel models."""
        stats = {}
        for name, loaded_model in model_manager.models.items():
            if loaded_model.is_data_parallel and loaded_model.data_parallel_engine:
                stats[name] = loaded_model.data_parallel_engine.get_stats()
        
        if not stats:
            return {"message": "No data-parallel models enabled"}
        
        return {
            "data_parallel_engines": stats,
            "total_models": len(stats),
        }
    
    @app.get("/v1/models")
    async def list_models():
        return model_manager.list_models()
    
    @app.post("/v1/chat/completions", response_model=None)
    async def create_chat_completion(
        request: ChatCompletionRequest,
    ):
        """Chat completion endpoint (multi-model)."""
        import time
        start_time = time.time()
        
        try:
            requested_model = request.model
            
            if not model_manager.model_exists(requested_model):
                if model_manager.default_model:
                    print(f"Model '{requested_model}' not found, using default '{model_manager.default_model}'")
                    requested_model = model_manager.default_model
                else:
                    error = model_manager.create_model_not_found_error(requested_model)
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content=error.model_dump(),
                    )
            
            chat_service = model_manager.get_chat_service(requested_model)
            if chat_service is None:
                error = model_manager.create_model_not_found_error(requested_model)
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content=error.model_dump(),
                )
            
            result = await chat_service.create_chat_completion(request)
            
            if isinstance(result, ErrorResponse):
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content=result.model_dump(),
                )
            
            if hasattr(result, "__aiter__"):
                if request_logger:
                    async def logged_chat_stream():
                        chunks = []
                        async for chunk in result:
                            chunks.append(chunk)
                            yield chunk
                        duration_ms = (time.time() - start_time) * 1000
                        try:
                            request_logger.log_chat_completion_stream(
                                request=request.model_dump(),
                                chunks=chunks,
                                model=requested_model,
                                duration_ms=duration_ms,
                            )
                        except Exception as log_error:
                            print(f"Failed to log streaming response: {log_error}")
                    
                    return StreamingResponse(
                        logged_chat_stream(),
                        media_type="text/event-stream",
                    )
                else:
                    return StreamingResponse(
                        result,
                        media_type="text/event-stream",
                    )
            
            if request_logger:
                duration_ms = (time.time() - start_time) * 1000
                try:
                    request_logger.log_chat_completion(
                        request=request.model_dump(),
                        response=result.model_dump(),
                        model=requested_model,
                        duration_ms=duration_ms,
                    )
                except Exception as log_error:
                    print(f"Failed to log response: {log_error}")
            
            return result
        
        except Exception as e:
            error = ErrorResponse(
                message=f"Internal server error: {str(e)}",
                type="InternalServerError",
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error.model_dump(),
            )
    
    @app.post("/v1/completions", response_model=None)
    async def create_completion(
        request: CompletionRequest,
    ):
        """Text completion endpoint (multi-model)."""
        import time
        start_time = time.time()
        
        try:
            requested_model = request.model
            
            if not model_manager.model_exists(requested_model):
                if model_manager.default_model:
                    print(f"Model '{requested_model}' not found, using default '{model_manager.default_model}'")
                    requested_model = model_manager.default_model
                else:
                    error = model_manager.create_model_not_found_error(requested_model)
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content=error.model_dump(),
                    )
            
            completion_service = model_manager.get_completion_service(requested_model)
            if completion_service is None:
                error = model_manager.create_model_not_found_error(requested_model)
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content=error.model_dump(),
                )
            
            result = await completion_service.create_completion(request)
            
            if isinstance(result, ErrorResponse):
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content=result.model_dump(),
                )
            
            if hasattr(result, "__aiter__"):
                if request_logger:
                    async def logged_stream():
                        chunks = []
                        async for chunk in result:
                            chunks.append(chunk)
                            yield chunk
                        duration_ms = (time.time() - start_time) * 1000
                        try:
                            request_logger.log_completion_stream(
                                request=request.model_dump(),
                                chunks=chunks,
                                model=requested_model,
                                duration_ms=duration_ms,
                            )
                        except Exception as log_error:
                            print(f"Failed to log streaming response: {log_error}")
                    
                    return StreamingResponse(
                        logged_stream(),
                        media_type="text/event-stream",
                    )
                else:
                    return StreamingResponse(
                        result,
                        media_type="text/event-stream",
                    )
            
            if request_logger:
                duration_ms = (time.time() - start_time) * 1000
                try:
                    request_logger.log_completion(
                        request=request.model_dump(),
                        response=result.model_dump(),
                        model=requested_model,
                        duration_ms=duration_ms,
                    )
                except Exception as log_error:
                    print(f"Failed to log response: {log_error}")
            
            return result
        
        except Exception as e:
            error = ErrorResponse(
                message=f"Internal server error: {str(e)}",
                type="InternalServerError",
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error.model_dump(),
            )
    
    return app


def main():
    """Command-line entry point (legacy single-model interface)."""
    parser = argparse.ArgumentParser(
        description="RWKVServe OpenAI-Compatible API Server"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Model directory containing model.pth and tokenizer_config.json",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device (cuda/cpu), auto-detected by default",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="rwkv-7",
        help="Model name used in the API",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes",
    )
    
    args = parser.parse_args()
    
    app = create_app(
        model_path=args.model_path,
        device=args.device,
        model_name=args.model_name,
    )
    
    print(f"\nStarting API server: http://{args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    print(f"Model: {args.model_name}")
    print()
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
