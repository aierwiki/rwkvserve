"""
RWKV Inference Engine - Continuous Batching with State Cache

Designed for async API serving, providing:
1. Continuous Batching: Multiple requests share GPU compute
2. Chunked Prefill: Controls peak memory usage for long prompts
3. State Cache integration: Accelerates repeated/similar requests

Uses SchedulerCore as the underlying scheduler, sharing scheduling logic
with BatchGenerator and LLM.
"""

import torch
import asyncio
from typing import List, Dict, Optional, Any, TYPE_CHECKING
from rwkvserve.inference.core import TokenSampler
from rwkvserve.inference.scheduler_core import SchedulerCore, SequenceState, SeqStatus

if TYPE_CHECKING:
    from rwkvserve.inference.state_cache import StateCache


class RWKVEngine:
    """
    RWKV Inference Engine with Continuous Batching and State Cache.

    Uses SchedulerCore as the underlying scheduler, sharing scheduling logic
    with BatchGenerator and LLM.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size: int = 16,
        max_prefill_tokens: int = 4096,
        prefill_chunk_size: int = 512,
        state_cache: Optional["StateCache"] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        
        self.max_batch_size = max_batch_size
        self.prefill_chunk_size = prefill_chunk_size
        
        self.scheduler_core = SchedulerCore(
            model=model,
            tokenizer=tokenizer,
            max_batch_size=max_batch_size,
            max_batch_tokens=max_prefill_tokens,
            prefill_chunk_size=prefill_chunk_size,
            state_cache=state_cache,
        )
        
        self.state_cache = state_cache
        if state_cache is not None:
            print(f"RWKVEngine: State Cache enabled")
        
        self.state_pool = self.scheduler_core.state_pool
        
        self.n_layer = self.scheduler_core.n_layer
        self.n_embd = self.scheduler_core.n_embd
        self.n_head = self.scheduler_core.n_head
        self.head_size = self.scheduler_core.head_size
        
        self._running_loop = False
        
        self._request_queues: Dict[str, asyncio.Queue] = {}

    async def add_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: Dict[str, Any],
    ) -> asyncio.Queue:
        """
        Add a request to the engine.
        
        Returns:
            asyncio.Queue where generated tokens will be pushed.
        """
        token_ids = self.tokenizer.encode(prompt)
        
        cached_prefix_len = 0
        cached_state = None
        cached_output = None
        
        if self.state_cache is not None:
            cached_prefix_len, cached_state, cached_output = self.state_cache.lookup(token_ids)
        
        suffix_token_ids = token_ids[cached_prefix_len:] if cached_prefix_len > 0 else token_ids
        
        stop = sampling_params.get("stop", [])
        if isinstance(stop, str):
            stop = [stop]
        elif stop is None:
            stop = []
        
        seq = SequenceState(
            seq_id=request_id,
            prompt_tokens=suffix_token_ids,
            max_new_tokens=sampling_params.get("max_tokens", 100),
            temperature=sampling_params.get("temperature", 1.0),
            top_p=sampling_params.get("top_p", 1.0),
            top_k=sampling_params.get("top_k", 0),
            logit_bias=sampling_params.get("logit_bias"),
            stop=stop,
            eos_token_id=self.tokenizer.eos_token_id,
            enforcer=sampling_params.get("enforcer"),
        )
        
        seq.original_tokens = token_ids
        
        if cached_prefix_len > 0:
            seq.cached_prefix_len = cached_prefix_len
            seq.cached_state = cached_state
            seq.cached_output = cached_output
        
        output_queue = asyncio.Queue()
        seq.output_queue = output_queue
        self._request_queues[request_id] = output_queue
        
        self.scheduler_core.add_sequence(seq)
        
        return output_queue

    async def start_background_loop(self):
        """Start the background scheduling loop."""
        self._running_loop = True
        while self._running_loop:
            await self.step()
            await asyncio.sleep(0.001)
    
    def stop_background_loop(self):
        """Stop the background scheduling loop."""
        self._running_loop = False

    async def step(self):
        """Execute one scheduling step."""
        if not self.scheduler_core.has_pending_work():
            return

        with torch.no_grad():
            decode_batch, prefill_batch = self.scheduler_core._schedule()
            
            if decode_batch:
                await self._run_decode_async(decode_batch)
            
            if prefill_batch:
                await self._run_prefill_async(prefill_batch)

    async def _run_decode_async(self, decode_batch: List[SequenceState]):
        """Run decode step and push tokens to output queues."""
        indices = [s.slot_id for s in decode_batch]
        att_x, att_kv, ffn_x = self.scheduler_core._gather_states(indices)
        
        input_tensor = torch.tensor(
            [[s.last_token_id] for s in decode_batch],
            dtype=torch.long, device=self.device
        )
        
        final_x, new_states = self.scheduler_core._model_forward_with_state(
            input_tensor, att_x, att_kv, ffn_x
        )
        self.scheduler_core._scatter_states(indices, new_states)
        
        logits = self.model.head(final_x[:, -1, :]).to(torch.float32)
        
        for i, seq in enumerate(decode_batch):
            seq_logits = logits[i]
            
            if seq.enforcer is not None and not seq.enforcer_finished:
                try:
                    seq_logits = seq.enforcer.mask_logits(seq_logits)
                except Exception as e:
                    print(f"Enforcer mask_logits failed: {e}")
            
            token_id, _ = TokenSampler.sample(
                seq_logits,
                temperature=seq.temperature,
                top_p=seq.top_p,
                top_k=seq.top_k,
                logit_bias=seq.logit_bias
            )
            
            if seq.enforcer is not None and not seq.enforcer_finished:
                try:
                    seq.enforcer_finished = seq.enforcer.update(token_id)
                except Exception as e:
                    print(f"Enforcer update failed: {e}")
                    seq.enforcer_finished = True
            
            seq.last_token_id = token_id
            seq.generated_tokens.append(token_id)
            
            token_text = self.tokenizer.decode([token_id])
            seq.generated_text += token_text

            # Incremental stop check (byte-safe; truncates text to OpenAI spec)
            stop_detected = self.scheduler_core._post_sample(seq, token_id)
            if stop_detected and seq.stop_checker is not None:
                seq.generated_text = seq.stop_checker.keep_text

            if seq.output_queue is not None:
                await seq.output_queue.put(token_id)

            if seq.is_finished or stop_detected:
                self.scheduler_core._finish_sequence(seq)
                if seq.output_queue is not None:
                    await seq.output_queue.put(None)

    async def _run_prefill_async(self, prefill_batch: List[tuple]):
        """Run prefill step for queued sequences."""
        import collections
        
        jobs_by_len = collections.defaultdict(list)
        for seq, chunk in prefill_batch:
            jobs_by_len[len(chunk)].append((seq, chunk))
        
        for length, jobs in jobs_by_len.items():
            indices = [j[0].slot_id for j in jobs]
            chunks = [j[1] for j in jobs]
            
            att_x, att_kv, ffn_x = self.scheduler_core._gather_states(indices)
            input_tensor = torch.tensor(chunks, dtype=torch.long, device=self.device)
            
            final_x, new_states = self.scheduler_core._model_forward_with_state(
                input_tensor, att_x, att_kv, ffn_x
            )
            self.scheduler_core._scatter_states(indices, new_states)
            
            for i, (seq, chunk) in enumerate(jobs):
                seq.prefill_offset += len(chunk)
                
                if seq.prefill_offset >= len(seq.prompt_tokens):
                    seq.status = SeqStatus.RUNNING
                    if seq in self.scheduler_core.prefilling_seqs:
                        self.scheduler_core.prefilling_seqs.remove(seq)
                    self.scheduler_core.running_seqs.append(seq)
                    
                    seq_logits = self.model.head(final_x[i, -1, :]).to(torch.float32)
                    
                    if seq.enforcer is not None and not seq.enforcer_finished:
                        try:
                            seq_logits = seq.enforcer.mask_logits(seq_logits)
                        except Exception as e:
                            print(f"Enforcer mask_logits failed: {e}")
                    
                    token_id, _ = TokenSampler.sample(
                        seq_logits,
                        temperature=seq.temperature,
                        top_p=seq.top_p,
                        top_k=seq.top_k,
                        logit_bias=seq.logit_bias
                    )
                    
                    if seq.enforcer is not None and not seq.enforcer_finished:
                        try:
                            seq.enforcer_finished = seq.enforcer.update(token_id)
                        except Exception as e:
                            print(f"Enforcer update failed: {e}")
                            seq.enforcer_finished = True
                    
                    seq.last_token_id = token_id
                    seq.generated_tokens.append(token_id)
                    
                    token_text = self.tokenizer.decode([token_id])
                    seq.generated_text += token_text

                    stop_detected = self.scheduler_core._post_sample(seq, token_id)
                    if stop_detected and seq.stop_checker is not None:
                        seq.generated_text = seq.stop_checker.keep_text

                    if seq.output_queue is not None:
                        await seq.output_queue.put(token_id)

                    if seq.is_finished or stop_detected:
                        self.scheduler_core._finish_sequence(seq)
                        if seq.output_queue is not None:
                            await seq.output_queue.put(None)

    def _model_forward_with_state(self, input_ids, att_x_prevs, att_kvs, ffn_x_prevs):
        """Delegate to SchedulerCore."""
        return self.scheduler_core._model_forward_with_state(
            input_ids, att_x_prevs, att_kvs, ffn_x_prevs
        )
