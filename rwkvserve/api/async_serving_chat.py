"""
Async Chat Completion Service

Uses RWKVEngine for Continuous Batching, supporting:
- n>1: Generate multiple candidate replies simultaneously
- Streaming: Token-by-token delivery
- n>1 + streaming: Parallel sequences streamed by index
- State Cache: Accelerate similar requests
- response_format: JSON Schema structured output constraints
"""

import asyncio
from typing import AsyncGenerator, List, Optional, Union, Dict, Any, TYPE_CHECKING

from rwkvserve.api.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    ChatMessage,
    DeltaMessage,
    ErrorResponse,
    UsageInfo,
)
from rwkvserve.api.serving_engine import ServingEngine
from rwkvserve.engine.core import RWKVEngine
from rwkvserve.inference.structured_output import (
    create_enforcer,
    is_structured_output_available,
)

if TYPE_CHECKING:
    from rwkvserve.inference.state_cache import StateCache
    from rwkvserve.reasoning.base import ReasoningParser


class AsyncChatCompletionService(ServingEngine):
    """
    Async chat completion service with Continuous Batching,
    n>1 support, streaming, State Cache, structured output,
    and reasoning content extraction (vLLM-compatible).
    """
    
    def __init__(
        self,
        engine: RWKVEngine,
        *args,
        state_cache: Optional["StateCache"] = None,
        reasoning_parser: Optional["ReasoningParser"] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.engine = engine
        self.state_cache = state_cache
        self.reasoning_parser = reasoning_parser
        
    async def start(self):
        """Start the background inference loop."""
        if not self.engine._running_loop:
            asyncio.create_task(self.engine.start_background_loop())

    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert a list of chat messages to a prompt string."""
        messages_dict = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        try:
            return self.tokenizer.apply_chat_template(
                messages_dict,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"Warning: apply_chat_template failed: {e}, using default format")
            prompt_parts = []
            for msg in messages:
                if msg.role == "system":
                    prompt_parts.append(f"System: {msg.content}\n\n")
                elif msg.role == "user":
                    prompt_parts.append(f"User: {msg.content}\n\n")
                elif msg.role == "assistant":
                    prompt_parts.append(f"Assistant: {msg.content}\n\n")
            prompt_parts.append("Assistant:")
            return "".join(prompt_parts)

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> Union[ChatCompletionResponse, AsyncGenerator[str, None], ErrorResponse]:
        """
        Create a chat completion.

        Supports n=1, n>1, streaming, and response_format (JSON Schema).
        """
        error = self.validate_request_params(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        if error:
            return error
        
        request_id = self.generate_request_id()
        prompt = self._messages_to_prompt(request.messages)
        n = request.n if request.n and request.n > 0 else 1
        
        enforcer = None
        if request.response_format is not None:
            response_format_dict = request.response_format.model_dump()
            if response_format_dict.get("type") != "text":
                if not is_structured_output_available():
                    return self.create_error_response(
                        message="Structured output requires lm-format-enforcer. Install with: pip install lm-format-enforcer",
                        error_type="BadRequestError",
                    )
                try:
                    enforcer = create_enforcer(self.tokenizer, response_format_dict)
                except Exception as e:
                    return self.create_error_response(
                        message=f"Failed to create JSON Schema enforcer: {str(e)}",
                        error_type="BadRequestError",
                    )
        
        if enforcer is not None and n > 1:
            return self.create_error_response(
                message="response_format with n>1 is not supported. Please use n=1 with response_format.",
                error_type="BadRequestError",
            )
        
        sampling_params = {
            "max_tokens": request.max_tokens or 1024,
            "temperature": request.temperature if request.temperature is not None else 1.0,
            "top_p": request.top_p if request.top_p is not None else 1.0,
            "top_k": request.top_k if request.top_k is not None else 0,
            "logit_bias": request.logit_bias,
            "stop": request.stop,
            "enforcer": enforcer,
        }
        
        queues: List[asyncio.Queue] = []
        for i in range(n):
            seq_id = f"{request_id}-{i}" if n > 1 else request_id
            queue = await self.engine.add_request(seq_id, prompt, sampling_params)
            queues.append(queue)
        
        if request.stream:
            return self._stream_response_n(request_id, queues, request.model, n)
        else:
            return await self._collect_response_n(request_id, queues, request.model, n, prompt)

    async def _stream_response_n(
        self, 
        request_id: str, 
        queues: List[asyncio.Queue], 
        model_name: str,
        n: int
    ) -> AsyncGenerator[str, None]:
        """
        Stream response generator supporting n>1 and reasoning_content.

        Each chunk's choices array contains deltas for all active sequences.
        When reasoning is enabled, the delta will have reasoning_content
        during the thinking phase and content after </think>.
        """
        initial_choices = [
            ChatCompletionStreamResponseChoice(
                index=i,
                delta=DeltaMessage(role="assistant", content=""),
                finish_reason=None
            )
            for i in range(n)
        ]
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            model=model_name,
            choices=initial_choices
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

        finished = [False] * n
        # Per-sequence state for streaming reasoning parser
        accumulated_text = [""] * n
        accumulated_ids: List[List[int]] = [[] for _ in range(n)]
        
        while not all(finished):
            tasks = []
            active_indices = []
            
            for i, queue in enumerate(queues):
                if not finished[i]:
                    tasks.append(asyncio.create_task(queue.get()))
                    active_indices.append(i)
            
            if not tasks:
                break
            
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            choices = []
            for task in done:
                task_idx = tasks.index(task)
                seq_idx = active_indices[task_idx]
                
                token_id = task.result()
                
                if token_id is None:
                    finished[seq_idx] = True
                    choices.append(
                        ChatCompletionStreamResponseChoice(
                            index=seq_idx,
                            delta=DeltaMessage(content=""),
                            finish_reason="stop"
                        )
                    )
                else:
                    token = self.tokenizer.decode([token_id])

                    if self.reasoning_parser is not None:
                        prev_text = accumulated_text[seq_idx]
                        prev_ids = accumulated_ids[seq_idx]
                        accumulated_text[seq_idx] += token
                        accumulated_ids[seq_idx].append(token_id)

                        reasoning_delta, content_delta = (
                            self.reasoning_parser.extract_reasoning_content_streaming(
                                previous_text=prev_text,
                                current_text=accumulated_text[seq_idx],
                                delta_text=token,
                                previous_token_ids=prev_ids,
                                current_token_ids=accumulated_ids[seq_idx],
                                delta_token_ids=[token_id],
                            )
                        )

                        choices.append(
                            ChatCompletionStreamResponseChoice(
                                index=seq_idx,
                                delta=DeltaMessage(
                                    content=content_delta,
                                    reasoning_content=reasoning_delta,
                                ),
                                finish_reason=None
                            )
                        )
                    else:
                        choices.append(
                            ChatCompletionStreamResponseChoice(
                                index=seq_idx,
                                delta=DeltaMessage(content=token),
                                finish_reason=None
                            )
                        )
            
            if choices:
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    model=model_name,
                    choices=choices
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
            
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        yield "data: [DONE]\n\n"

    async def _collect_response_n(
        self, 
        request_id: str, 
        queues: List[asyncio.Queue], 
        model_name: str,
        n: int,
        prompt: str
    ) -> ChatCompletionResponse:
        """Collect complete responses from all n sequences in parallel."""
        async def collect_one(queue: asyncio.Queue) -> List[int]:
            tokens = []
            while True:
                token_id = await queue.get()
                if token_id is None:
                    break
                tokens.append(token_id)
            return tokens
        
        all_tokens = await asyncio.gather(*[collect_one(q) for q in queues])
        
        choices = []
        total_completion_tokens = 0
        
        for i, tokens in enumerate(all_tokens):
            text = self.tokenizer.decode(tokens)
            total_completion_tokens += len(tokens)

            reasoning_content = None
            content = text
            if self.reasoning_parser is not None:
                reasoning_content, content = self.reasoning_parser.extract_reasoning_content(text)

            choices.append(
                ChatCompletionResponseChoice(
                    index=i,
                    message=ChatMessage(
                        role="assistant",
                        content=content,
                        reasoning_content=reasoning_content,
                    ),
                    finish_reason="stop"
                )
            )
        
        prompt_tokens = len(self.tokenizer.encode(prompt))
        
        return ChatCompletionResponse(
            id=request_id,
            model=model_name,
            choices=choices,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=prompt_tokens + total_completion_tokens
            )
        )
