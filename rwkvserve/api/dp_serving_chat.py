"""
Data-Parallel Chat Service - Wraps DataParallelEngine

Provides the same interface as AsyncChatCompletionService while
using DataParallelEngine for multi-GPU parallel inference,
transparent to the upper API layer.
"""

import asyncio
from typing import Any, Optional, Dict, List, AsyncGenerator

from rwkvserve.api.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    UsageInfo,
)


class DataParallelChatService:
    """
    Data-parallel chat service wrapping DataParallelEngine.

    Provides the same interface as other ChatService implementations.
    """
    
    def __init__(
        self,
        engine: "DataParallelEngine",
        tokenizer: Any,
        model_name: str,
    ):
        self.engine = engine
        self.tokenizer = tokenizer
        self.model_name = model_name
        self._started = False
    
    async def ensure_started(self):
        """Ensure the engine has been started."""
        if not self._started:
            success = await self.engine.start()
            if not success:
                raise RuntimeError("Data-parallel engine failed to start")
            self._started = True
    
    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Create a non-streaming chat completion."""
        await self.ensure_started()
        
        request_dict = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens or 1024,
            "stream": False,
        }
        
        if request.stop:
            request_dict["stop"] = request.stop
        if request.presence_penalty:
            request_dict["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty:
            request_dict["frequency_penalty"] = request.frequency_penalty
        
        result = await self.engine.generate(**request_dict)
        
        if isinstance(result, dict):
            return ChatCompletionResponse(**result)
        elif isinstance(result, ChatCompletionResponse):
            return result
        
        content = str(result)
        prompt = self.tokenizer.apply_chat_template(
            self._format_messages(request.messages),
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_tokens = len(self.tokenizer.encode(prompt))
        completion_tokens = len(self.tokenizer.encode(content))
        
        return ChatCompletionResponse(
            id=f"chatcmpl-dp-{id(result)}",
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
    
    async def create_chat_completion_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[ChatCompletionStreamResponse, None]:
        """Create a streaming chat completion."""
        await self.ensure_started()
        
        request_dict = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens or 1024,
            "stream": True,
        }
        
        if request.stop:
            request_dict["stop"] = request.stop
        if request.presence_penalty:
            request_dict["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty:
            request_dict["frequency_penalty"] = request.frequency_penalty
        
        async for chunk in self.engine.generate_stream(**request_dict):
            if isinstance(chunk, dict):
                yield ChatCompletionStreamResponse(**chunk)
            elif isinstance(chunk, ChatCompletionStreamResponse):
                yield chunk
            else:
                content = str(chunk)
                yield ChatCompletionStreamResponse(
                    id=f"chatcmpl-dp-stream",
                    model=request.model,
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=0,
                            delta=DeltaMessage(content=content),
                            finish_reason=None,
                        )
                    ],
                )
    
    def _format_messages(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
    
    async def shutdown(self):
        """Shut down the engine."""
        if self._started:
            await self.engine.shutdown()
            self._started = False
