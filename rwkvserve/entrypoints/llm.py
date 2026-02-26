"""
LLM class - Continuous Batching implementation

Aligned with vLLM architecture:
- Uses SchedulerCore for all request handling
- Request-level progress tracking
- Dynamic scheduling (short requests are not blocked by long ones)

Example:
    >>> from rwkvserve.entrypoints.llm import LLM
    >>> from rwkvserve import SamplingParams
    >>>
    >>> llm = LLM(model="/path/to/model")
    >>> params = SamplingParams(temperature=0.8, max_tokens=100)
    >>> outputs = llm.generate(["Hello"] * 10000, params, use_tqdm=True)
"""

import os
import time
import torch
from typing import List, Optional, Union, Dict, Any, Callable
from dataclasses import dataclass

from rwkvserve.sampling_params import SamplingParams
from rwkvserve.outputs import RequestOutput, CompletionOutput


@dataclass
class RequestState:
    """Tracks completion state for a single request."""
    request_id: int
    prompt: str
    prompt_token_ids: List[int]
    sampling_params: SamplingParams
    
    generated_tokens: List[int]
    finished: bool = False
    finish_reason: Optional[str] = None


class LLM:
    """
    Offline batch inference LLM with Continuous Batching.

    Uses SchedulerCore for dynamic scheduling: short requests are never
    blocked by long ones, and an arbitrary number of requests can be
    processed in a single call.

    Args:
        model: Path to model directory.
        device: Inference device ("cuda", "cpu").
        dtype: Model precision ("bf16", "fp16", "fp32").
        gpu_id: GPU device ID.
        max_batch_size: Maximum concurrent sequences for the scheduler (default: 256).
        max_batch_tokens: Maximum tokens per forward pass (default: 8192).
        prefill_chunk_size: Chunked prefill block size (default: 512).
        use_state_cache: Whether to enable State Cache (default: False).
        state_cache_memory_gb: State Cache memory limit in GB (default: 4.0).

    Example:
        >>> llm = LLM(model="/path/to/model", max_batch_size=256)
        >>> outputs = llm.generate([...10000 prompts...], SamplingParams())
    """
    
    def __init__(
        self,
        model: str,
        device: str = "cuda",
        dtype: str = "bf16",
        gpu_id: Optional[int] = None,
        max_batch_size: int = 256,
        max_batch_tokens: int = 8192,
        prefill_chunk_size: int = 512,
        use_state_cache: bool = False,
        state_cache_memory_gb: float = 4.0,
        enable_lora: bool = False,
        lora_path: Optional[str] = None,
    ):
        """Initialize the LLM with Continuous Batching support."""
        if gpu_id is not None and device.startswith("cuda"):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            device = "cuda"
        
        self.device = device
        self.dtype = dtype
        self.model_path = model
        
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.prefill_chunk_size = prefill_chunk_size
        self.use_state_cache = use_state_cache
        self.state_cache_memory_gb = state_cache_memory_gb
        
        from rwkvserve import AutoModel, AutoTokenizer
        
        print(f"[LLM] Continuous Batching mode")
        print(f"[LLM] Loading model: {model}")
        print(f"[LLM] Device: {device}, dtype: {dtype}")
        print(f"[LLM] Scheduler config: max_batch_size={max_batch_size}, "
              f"max_batch_tokens={max_batch_tokens}")
        
        debug_mem = os.environ.get("RWKV_DEBUG_MEMORY", "0") == "1"
        if debug_mem and torch.cuda.is_available():
            print(f"[MEM] Before model load: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model, device=device, dtype=dtype)

        if enable_lora and lora_path:
            from rwkvserve.peft import load_and_merge_lora
            print(f"[LLM] Loading LoRA adapter: {lora_path}")
            self.model, _ = load_and_merge_lora(self.model, lora_path)

        self.model.eval()
        
        if debug_mem and torch.cuda.is_available():
            print(f"[MEM] After model load: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        self.eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
        
        self.state_cache = None
        if use_state_cache:
            from rwkvserve.inference.state_cache import StateCache, CacheConfig, CacheLevel
            
            n_layer = self.model.config.n_layer
            n_embd = self.model.config.n_embd
            head_size = getattr(self.model.config, 'head_size_a', 64)
            dim_att = getattr(self.model.config, 'dim_att', n_embd)
            n_head = dim_att // head_size
            vocab_size = getattr(self.model.config, 'vocab_size', 65536)
            
            state_size_bytes = (
                n_layer * n_embd * 2 +                              # att_x_prev
                n_layer * n_head * head_size * head_size * 4 +      # att_kv
                n_layer * n_embd * 2 +                              # ffn_x_prev
                vocab_size * 4                                       # output
            )
            
            max_memory_bytes = int(state_cache_memory_gb * 1e9 * 0.9)
            max_entries = max(1, max_memory_bytes // state_size_bytes)
            
            config = CacheConfig(
                max_cache_memory_gb=state_cache_memory_gb,
                max_cache_entries=max_entries,
                cache_level=CacheLevel.PREFIX,
                cache_first_message=True,
            )
            
            self.state_cache = StateCache(self.model, config=config, auto_config=False)
            
            print(f"[LLM] State Cache enabled "
                  f"(memory limit: {state_cache_memory_gb}GB, max entries: {max_entries})")
        
        # Pre-create SchedulerCore to avoid GPU memory fragmentation across generate() calls
        from rwkvserve.inference.scheduler_core import SchedulerCore
        self._scheduler = SchedulerCore(
            model=self.model,
            tokenizer=self.tokenizer,
            max_batch_size=self.max_batch_size,
            max_batch_tokens=self.max_batch_tokens,
            prefill_chunk_size=self.prefill_chunk_size,
            state_cache=self.state_cache,
        )
        
        print(f"[LLM] Model loaded successfully")
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        use_tqdm: bool = True,
        on_request_complete: Optional[Callable[[int, RequestOutput], None]] = None,
        on_token_generated: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[RequestOutput]:
        """
        Generate text for a batch of prompts using Continuous Batching.

        Args:
            prompts: A single prompt string or list of prompts (any quantity).
            sampling_params: Sampling parameters for generation.
            use_tqdm: Whether to display a progress bar.
            on_request_complete: Callback invoked when a request finishes.
                Signature: (request_id, output) -> None.
            on_token_generated: Callback invoked for each generated token.
                Signature: (request_id, token_id, token_text) -> None.

        Returns:
            List[RequestOutput]: Generation results in input order.

        Example:
            >>> llm = LLM(model="...")
            >>> outputs = llm.generate(["Hello"] * 10000, use_tqdm=True)
            >>>
            >>> def save_result(req_id, output):
            ...     print(f"Request {req_id} done: {output.outputs[0].text[:50]}...")
            >>> outputs = llm.generate(prompts, on_request_complete=save_result)
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        return self._generate_continuous_batching(
            prompts, sampling_params, use_tqdm,
            on_request_complete=on_request_complete,
            on_token_generated=on_token_generated,
        )
    
    def _generate_continuous_batching(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        use_tqdm: bool,
        on_request_complete: Optional[Callable[[int, RequestOutput], None]] = None,
        on_token_generated: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[RequestOutput]:
        """
        Run Continuous Batching over all prompts.

        Reuses the pre-created SchedulerCore to avoid GPU memory fragmentation.
        Dynamically schedules sequences and tracks per-request progress.
        """
        from tqdm import tqdm
        from rwkvserve.inference.scheduler_core import SequenceState
        
        total_requests = len(prompts)
        
        pbar = None
        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0
        if use_tqdm:
            pbar = tqdm(
                total=total_requests,
                desc="Processed prompts",
                unit="prompt",
                dynamic_ncols=True,
                postfix=f"in: 0 toks/s, out: 0 toks/s",
            )
        
        print(f"\n[Continuous Batching] Processing {total_requests} request(s)")
        print(f"  Scheduler config: max_batch_size={self.max_batch_size}, "
              f"max_batch_tokens={self.max_batch_tokens}")
        
        scheduler = self._scheduler
        scheduler.reset()
        
        request_states: Dict[int, RequestState] = {}
        for i, prompt in enumerate(prompts):
            prompt_token_ids = self.tokenizer.encode(prompt)
            request_states[i] = RequestState(
                request_id=i,
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                generated_tokens=[],
            )
        
        sequences = []
        for i, prompt in enumerate(prompts):
            prompt_tokens = request_states[i].prompt_token_ids
            seq = SequenceState(
                seq_id=i,
                prompt_tokens=prompt_tokens,
                max_new_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                top_k=sampling_params.top_k,
                logit_bias=sampling_params.logit_bias,
                eos_token_id=self.eos_token_id,
            )
            sequences.append(seq)
        
        scheduler.add_sequences(sequences)
        
        step_count = 0
        finished_count = 0
        
        for req_state in request_states.values():
            total_input_tokens += len(req_state.prompt_token_ids)
        
        import torch
        with torch.no_grad():
            while scheduler.has_pending_work():
                step_count += 1
                
                step_outputs = scheduler.step_sync()
                
                for seq_id, token_id in step_outputs:
                    req_state = request_states[seq_id]
                    req_state.generated_tokens.append(token_id)
                    total_output_tokens += 1
                    
                    if on_token_generated is not None:
                        try:
                            token_text = self.tokenizer.decode([token_id])
                            on_token_generated(seq_id, token_id, token_text)
                        except Exception:
                            pass
                    
                    if self._is_request_finished(req_state, sequences[seq_id]):
                        if not req_state.finished:
                            req_state.finished = True
                            req_state.finish_reason = self._get_finish_reason(
                                req_state, sequences[seq_id]
                            )
                            finished_count += 1
                            
                            if on_request_complete is not None:
                                try:
                                    output = self._build_request_output(req_state)
                                    on_request_complete(seq_id, output)
                                except Exception:
                                    pass
                            
                            if pbar is not None:
                                elapsed = time.time() - start_time
                                if elapsed > 0:
                                    in_speed = total_input_tokens / elapsed
                                    out_speed = total_output_tokens / elapsed
                                    pbar.postfix = (
                                        f"in: {in_speed:.0f} toks/s, "
                                        f"out: {out_speed:.0f} toks/s"
                                    )
                                pbar.update(1)
        
        if pbar is not None:
            pbar.close()
        
        elapsed = time.time() - start_time
        in_speed = total_input_tokens / elapsed if elapsed > 0 else 0
        out_speed = total_output_tokens / elapsed if elapsed > 0 else 0
        print(f"\n[Continuous Batching] Completed {total_requests} request(s)")
        print(f"  Steps: {step_count}, elapsed: {elapsed:.2f}s")
        print(f"  Input: {total_input_tokens} tokens ({in_speed:.0f} toks/s)")
        print(f"  Output: {total_output_tokens} tokens ({out_speed:.0f} toks/s)")
        
        outputs = []
        for i in range(total_requests):
            req_state = request_states[i]
            outputs.append(self._build_request_output(req_state))
        
        return outputs
    
    def _is_request_finished(
        self,
        req_state: RequestState,
        seq_state: Any,
    ) -> bool:
        """Check whether a request has finished generating."""
        if seq_state.is_finished:
            return True
        
        if len(req_state.generated_tokens) >= req_state.sampling_params.max_tokens:
            return True
        
        if (self.eos_token_id is not None and 
            req_state.generated_tokens and 
            req_state.generated_tokens[-1] == self.eos_token_id):
            return True
        
        return False
    
    def _get_finish_reason(
        self,
        req_state: RequestState,
        seq_state: Any,
    ) -> str:
        """Determine the finish reason for a completed request."""
        if (self.eos_token_id is not None and 
            req_state.generated_tokens and 
            req_state.generated_tokens[-1] == self.eos_token_id):
            return "stop"
        
        if len(req_state.generated_tokens) >= req_state.sampling_params.max_tokens:
            return "length"
        
        return "stop"
    
    def _build_request_output(self, req_state: RequestState) -> RequestOutput:
        """Build a RequestOutput from the completed request state."""
        completion_ids = req_state.generated_tokens.copy()
        
        if (self.eos_token_id is not None and 
            completion_ids and 
            completion_ids[-1] == self.eos_token_id):
            completion_ids = completion_ids[:-1]
        
        completion_text = self.tokenizer.decode(completion_ids)
        
        completion_output = CompletionOutput(
            index=0,
            text=completion_text,
            token_ids=completion_ids,
            finish_reason=req_state.finish_reason or "stop",
        )
        
        return RequestOutput(
            request_id=f"req-{req_state.request_id}",
            prompt=req_state.prompt,
            prompt_token_ids=req_state.prompt_token_ids,
            outputs=[completion_output],
            finished=True,
        )


def create_llm_with_continuous_batching(
    model: str,
    device: str = "cuda",
    dtype: str = "bf16",
    **kwargs
) -> LLM:
    """Create an LLM instance with Continuous Batching support."""
    return LLM(model=model, device=device, dtype=dtype, **kwargs)
