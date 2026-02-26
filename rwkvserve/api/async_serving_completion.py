"""
Async Completion Service

Uses RWKVEngine for Continuous Batching, supporting:
- n>1: Generate multiple candidate completions simultaneously
- Streaming: Token-by-token delivery
- n>1 + streaming: Parallel sequences streamed by index
- State Cache: Accelerate similar requests
"""

import asyncio
from typing import AsyncGenerator, List, Optional, Union, Dict, Any, TYPE_CHECKING

from rwkvserve.api.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionStreamResponse,
    CompletionStreamResponseChoice,
    ErrorResponse,
    UsageInfo,
)
from rwkvserve.api.serving_engine import ServingEngine
from rwkvserve.engine.core import RWKVEngine

if TYPE_CHECKING:
    from rwkvserve.inference.state_cache import StateCache


class AsyncCompletionService(ServingEngine):
    """
    Async text completion service with Continuous Batching,
    n>1 support, streaming, and State Cache.
    """
    
    def __init__(self, engine: RWKVEngine, *args, state_cache: Optional["StateCache"] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = engine
        self.state_cache = state_cache
        
    async def start(self):
        """Start the background inference loop."""
        if not self.engine._running_loop:
            asyncio.create_task(self.engine.start_background_loop())

    async def create_completion(
        self,
        request: CompletionRequest,
    ) -> Union[CompletionResponse, AsyncGenerator[str, None], ErrorResponse]:
        """
        Create a text completion.

        Supports n=1, n>1, streaming, and n>1 + streaming.
        """
        error = self.validate_request_params(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        if error:
            return error
        
        if request.suffix:
            return self.create_error_response(
                message="suffix is not supported",
                param="suffix",
            )
        
        if request.best_of and request.best_of > 1:
            return self.create_error_response(
                message="best_of is not supported",
                param="best_of",
            )
            
        if isinstance(request.prompt, list):
            if len(request.prompt) > 1:
                return self.create_error_response(
                    message="Currently only single prompt is supported",
                    param="prompt",
                )
            prompt = request.prompt[0]
        else:
            prompt = request.prompt
            
        request_id = self.generate_request_id(prefix="cmpl")
        
        sampling_params = {
            "max_new_tokens": request.max_tokens or 16,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "logit_bias": request.logit_bias,
            "stop": request.stop,
        }
        
        if request.stream:
            return self._create_stream_response(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
                n=request.n,
                model_name=request.model,
            )
        
        return await self._create_full_response(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            n=request.n,
            model_name=request.model,
        )

    async def _create_stream_response(
        self,
        request_id: str,
        prompt: str,
        sampling_params: Dict[str, Any],
        n: int,
        model_name: str,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response."""
        queues = []
        for i in range(n):
            sub_request_id = f"{request_id}-{i}"
            queue = await self.engine.add_request(
                request_id=sub_request_id,
                prompt=prompt,
                sampling_params=sampling_params,
            )
            queues.append(queue)
            
        finished = [False] * n
        
        while not all(finished):
            for i, queue in enumerate(queues):
                if finished[i]:
                    continue
                    
                if not queue.empty():
                    item = await queue.get()
                    
                    if isinstance(item, Exception):
                        print(f"Error in request {request_id}-{i}: {item}")
                        finished[i] = True
                        continue
                        
                    if item is None:
                        finished[i] = True
                        continue
                        
                    token_text = self.tokenizer.decode([item]) if isinstance(item, int) else item
                    
                    choice = CompletionStreamResponseChoice(
                        index=i,
                        text=token_text,
                        finish_reason=None,
                    )
                    
                    response = CompletionStreamResponse(
                        id=request_id,
                        model=model_name,
                        choices=[choice],
                    )
                    
                    yield response.model_dump_json()
            
            await asyncio.sleep(0.001)
            
        for i in range(n):
            choice = CompletionStreamResponseChoice(
                index=i,
                text="",
                finish_reason="stop",
            )
            response = CompletionStreamResponse(
                id=request_id,
                model=model_name,
                choices=[choice],
            )
            yield response.model_dump_json()

    async def _create_full_response(
        self,
        request_id: str,
        prompt: str,
        sampling_params: Dict[str, Any],
        n: int,
        model_name: str,
    ) -> CompletionResponse:
        """Generate a complete (non-streaming) response."""
        queues = []
        for i in range(n):
            sub_request_id = f"{request_id}-{i}"
            queue = await self.engine.add_request(
                request_id=sub_request_id,
                prompt=prompt,
                sampling_params=sampling_params,
            )
            queues.append(queue)
            
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for i, queue in enumerate(queues):
            generated_text = ""
            token_ids = []
            finish_reason = "length"
            
            while True:
                item = await queue.get()
                
                if isinstance(item, Exception):
                    print(f"Error in request {request_id}-{i}: {item}")
                    break
                    
                if item is None:
                    finish_reason = "stop"
                    break

                token_ids.append(item) if isinstance(item, int) else None
                generated_text += self.tokenizer.decode([item]) if isinstance(item, int) else item

            completion_tokens = len(token_ids)
            total_completion_tokens += completion_tokens
            
            choice = CompletionResponseChoice(
                index=i,
                text=generated_text,
                finish_reason=finish_reason,
            )
            choices.append(choice)
            
        usage = UsageInfo(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        )
        
        return CompletionResponse(
            id=request_id,
            model=model_name,
            choices=choices,
            usage=usage,
        )
