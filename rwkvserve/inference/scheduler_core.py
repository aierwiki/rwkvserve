"""
Unified Scheduler Core - shared by RWKVEngine and BatchGenerator

Provides low-level Continuous Batching scheduling logic including slot management,
State Pool management, Prefill/Decode scheduling, Chunked Prefill, and optional
State Cache integration. Supports both synchronous and asynchronous execution modes.
"""

import torch
import collections
import os
import time
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto

DEBUG_MEMORY = os.environ.get("RWKV_DEBUG_MEMORY", "0") == "1"

def log_memory(tag: str):
    """Log GPU memory usage when DEBUG_MEMORY is enabled."""
    if DEBUG_MEMORY and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_alloc = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[MEM] {tag}: alloc={allocated:.2f}GB, reserved={reserved:.2f}GB, max={max_alloc:.2f}GB")

if TYPE_CHECKING:
    from rwkvserve.inference.state_cache import StateCache

class SeqStatus(Enum):
    """Sequence status"""
    WAITING = auto()
    PREFILLING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class SequenceState:
    """
    Unified sequence state.

    Supports both RWKVEngine (async serving) and BatchGenerator (offline batch).
    """
    seq_id: Union[int, str]
    prompt_tokens: List[int]
    status: SeqStatus = SeqStatus.WAITING

    prefill_offset: int = 0
    generated_tokens: List[int] = field(default_factory=list)
    last_token_id: Optional[int] = None

    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    logit_bias: Optional[dict] = None
    stop: List[str] = field(default_factory=list)
    eos_token_id: Optional[int] = None

    generated_text: str = ""

    slot_id: int = -1

    original_tokens: Optional[List[int]] = None
    cached_prefix_len: int = 0
    cached_state: Optional[Any] = None
    cached_output: Optional[torch.Tensor] = None

    output_queue: Optional[Any] = None

    enforcer: Optional[Any] = None
    enforcer_finished: bool = False

    @property
    def is_finished(self) -> bool:
        """Check whether the sequence has finished generation."""
        if self.enforcer is not None and self.enforcer_finished:
            return True
        if self.eos_token_id is not None and self.last_token_id == self.eos_token_id:
            return True
        if len(self.generated_tokens) >= self.max_new_tokens:
            return True
        return False


class SchedulerCore:
    """
    Unified scheduler core.

    Responsibilities:
    - Slot allocation and release
    - GPU State Pool management
    - Prefill/Decode scheduling
    - Chunked Prefill
    - Optional State Cache integration

    Usage:
    - BatchGenerator: uses step_sync() for synchronous execution
    - RWKVEngine: uses step_async() for asynchronous execution
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size: int = 64,
        max_batch_tokens: int = 8192,
        prefill_chunk_size: int = 512,
        state_cache: Optional["StateCache"] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.prefill_chunk_size = prefill_chunk_size

        self.state_cache = state_cache
        if state_cache is not None:
            print(f"SchedulerCore: State Cache integration enabled")

        self.n_layer = model.config.n_layer
        self.n_embd = model.config.n_embd
        self.head_size = getattr(model.config, 'head_size_a', 64)
        dim_att = getattr(model.config, 'dim_att', self.n_embd)
        self.n_head = getattr(model.config, 'n_head', dim_att // self.head_size)

        log_memory("Before State Pool init")
        self.state_pool = self._init_state_pool()
        log_memory("After State Pool init")

        self.slots: List[Optional[SequenceState]] = [None] * max_batch_size
        self.free_slots = collections.deque(range(max_batch_size))

        self.waiting_queue: collections.deque = collections.deque()
        self.prefilling_seqs: List[SequenceState] = []
        self.running_seqs: List[SequenceState] = []

        from rwkvserve.inference.core import TokenSampler
        self.sampler = TokenSampler

        # Ban out-of-vocabulary tokens during sampling
        if hasattr(tokenizer, 'idx2token') and tokenizer.idx2token:
            TokenSampler._max_valid_token_id = max(tokenizer.idx2token.keys())

    def _init_state_pool(self) -> Dict[str, torch.Tensor]:
        """
        Initialize the State Pool.

        RWKV7 state structure:
        - att_x_prev: [n_layer, max_batch, n_embd]
        - att_kv: [n_layer, max_batch, n_head, head_size, head_size]
        - ffn_x_prev: [n_layer, max_batch, n_embd]
        """
        state_dtype = self.dtype
        kv_dtype = torch.float32  # KV state kept in FP32 for precision

        return {
            "att_x_prev": torch.zeros(
                (self.n_layer, self.max_batch_size, self.n_embd),
                dtype=state_dtype, device=self.device
            ),
            "att_kv": torch.zeros(
                (self.n_layer, self.max_batch_size, self.n_head, self.head_size, self.head_size),
                dtype=kv_dtype, device=self.device
            ),
            "ffn_x_prev": torch.zeros(
                (self.n_layer, self.max_batch_size, self.n_embd),
                dtype=state_dtype, device=self.device
            ),
        }

    # =========================================================================
    # Public Interface
    # =========================================================================

    def reset(self) -> None:
        """Reset scheduler state while reusing the State Pool."""
        self.waiting_queue.clear()
        self.prefilling_seqs.clear()
        self.running_seqs.clear()

        self.slots = [None] * self.max_batch_size
        self.free_slots = collections.deque(range(self.max_batch_size))

    def add_sequences(self, sequences: List[SequenceState]) -> None:
        """Add multiple sequences to the waiting queue."""
        for seq in sequences:
            self.waiting_queue.append(seq)

    def add_sequence(self, seq: SequenceState) -> None:
        """Add a single sequence to the waiting queue."""
        self.waiting_queue.append(seq)

    def has_pending_work(self) -> bool:
        """Check whether there is pending work."""
        return bool(self.waiting_queue or self.prefilling_seqs or self.running_seqs)

    def get_stats(self) -> Dict[str, int]:
        """Get current scheduler statistics."""
        return {
            "waiting": len(self.waiting_queue),
            "prefilling": len(self.prefilling_seqs),
            "running": len(self.running_seqs),
            "free_slots": len(self.free_slots),
        }

    # =========================================================================
    # Core Scheduling Logic
    # =========================================================================

    def _schedule(self) -> Tuple[List[SequenceState], List[Tuple[SequenceState, List[int]]]]:
        """
        Schedule one step of work.

        Returns:
            decode_batch: sequences to decode
            prefill_batch: (sequence, chunk) pairs to prefill
        """
        current_budget = self.max_batch_tokens

        # 1. Assign slots from waiting queue
        while self.waiting_queue and self.free_slots:
            seq = self.waiting_queue.popleft()
            slot_id = self.free_slots.popleft()
            seq.slot_id = slot_id
            seq.status = SeqStatus.PREFILLING
            self.slots[slot_id] = seq
            self.prefilling_seqs.append(seq)

            self._init_slot_state(seq, slot_id)

            if seq.status == SeqStatus.RUNNING:
                self.prefilling_seqs.remove(seq)
                self.running_seqs.append(seq)

        # 2. Schedule decode (higher priority, 1 token per sequence)
        decode_batch: List[SequenceState] = []
        running_seqs_list = list(self.running_seqs)
        for seq in running_seqs_list:
            if len(decode_batch) < self.max_batch_size:
                decode_batch.append(seq)
            else:
                break

        # 3. Schedule prefill with remaining budget
        prefill_batch: List[Tuple[SequenceState, List[int]]] = []
        for seq in list(self.prefilling_seqs):
            if current_budget <= 0:
                break
            remaining = len(seq.prompt_tokens) - seq.prefill_offset
            chunk_len = min(remaining, self.prefill_chunk_size, current_budget)
            if chunk_len > 0:
                chunk = seq.prompt_tokens[seq.prefill_offset : seq.prefill_offset + chunk_len]
                prefill_batch.append((seq, chunk))
                current_budget -= chunk_len

        return decode_batch, prefill_batch

    def _init_slot_state(self, seq: SequenceState, slot_id: int) -> None:
        """
        Initialize slot state: query State Cache if enabled, load cached state
        or reset to zero.
        """
        if self.state_cache is not None and seq.prompt_tokens:
            prefix_len, state, output = self.state_cache.lookup(seq.prompt_tokens)
            if prefix_len > 0 and state is not None:
                seq.cached_state = state
                seq.cached_prefix_len = prefix_len
                seq.prefill_offset = prefix_len

        if seq.cached_state is not None:
            self._load_state_to_slot(slot_id, seq.cached_state)
            seq.cached_state = None

            if seq.prefill_offset >= len(seq.prompt_tokens):
                seq.status = SeqStatus.RUNNING
                seq.last_token_id = seq.prompt_tokens[-1]
        else:
            self._reset_slot_state(slot_id)

    def _reset_slot_state(self, slot_id: int) -> None:
        """Reset slot state to zero."""
        idx = torch.tensor([slot_id], device=self.device, dtype=torch.long)
        self.state_pool["att_x_prev"][:, idx, :] = 0
        self.state_pool["att_kv"][:, idx, ...] = 0
        self.state_pool["ffn_x_prev"][:, idx, :] = 0

    def _load_state_to_slot(self, slot_id: int, state: List[Dict[str, torch.Tensor]]) -> None:
        """Load CPU state into a GPU slot."""
        with torch.no_grad():
            for layer_idx, layer_state in enumerate(state):
                if 'att_x_prev' in layer_state:
                    self.state_pool["att_x_prev"][layer_idx, slot_id, :] = \
                        layer_state['att_x_prev'].to(self.device, dtype=self.dtype)
                if 'att_kv' in layer_state:
                    self.state_pool["att_kv"][layer_idx, slot_id, ...] = \
                        layer_state['att_kv'].to(self.device, dtype=torch.float32)
                if 'ffn_x_prev' in layer_state:
                    self.state_pool["ffn_x_prev"][layer_idx, slot_id, :] = \
                        layer_state['ffn_x_prev'].to(self.device, dtype=self.dtype)

    def _back_state_from_slot(self, slot_id: int) -> List[Dict[str, torch.Tensor]]:
        """Back up GPU slot state to CPU."""
        state = []
        with torch.no_grad():
            for layer_idx in range(self.n_layer):
                layer_state = {
                    'att_x_prev': self.state_pool["att_x_prev"][layer_idx, slot_id, :].cpu().clone(),
                    'att_kv': self.state_pool["att_kv"][layer_idx, slot_id, ...].cpu().clone(),
                    'ffn_x_prev': self.state_pool["ffn_x_prev"][layer_idx, slot_id, :].cpu().clone(),
                }
                state.append(layer_state)
        return state

    def _finish_sequence(self, seq: SequenceState) -> None:
        """Mark sequence as finished and release its slot."""
        seq.status = SeqStatus.FINISHED
        if seq in self.running_seqs:
            self.running_seqs.remove(seq)
        if seq in self.prefilling_seqs:
            self.prefilling_seqs.remove(seq)
        self.slots[seq.slot_id] = None
        self.free_slots.append(seq.slot_id)

    # =========================================================================
    # Execution Methods
    # =========================================================================

    def _gather_states(self, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather states for given slot indices."""
        idx_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)
        return (
            self.state_pool["att_x_prev"][:, idx_tensor, :],
            self.state_pool["att_kv"][:, idx_tensor, ...],
            self.state_pool["ffn_x_prev"][:, idx_tensor, :],
        )

    def _scatter_states(
        self,
        indices: List[int],
        new_states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """Scatter states back to pool."""
        idx_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)
        self.state_pool["att_x_prev"][:, idx_tensor, :] = new_states[0].to(dtype=self.dtype)
        self.state_pool["att_kv"][:, idx_tensor, ...] = new_states[1].to(dtype=torch.float32)
        self.state_pool["ffn_x_prev"][:, idx_tensor, :] = new_states[2].to(dtype=self.dtype)

    def _model_forward_with_state(
        self,
        input_ids: torch.Tensor,
        att_x_prevs: torch.Tensor,
        att_kvs: torch.Tensor,
        ffn_x_prevs: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Run model forward pass with state.

        Args:
            input_ids: [B, T]
            att_x_prevs: [L, B, C]
            att_kvs: [L, B, H, N, N]
            ffn_x_prevs: [L, B, C]

        Returns:
            final_x: [B, T, C] - last layer hidden states
            new_states: (att_x_prevs, att_kvs, ffn_x_prevs)
        """
        x = self.model.emb(input_ids)
        v_first = torch.empty_like(x)

        new_att_x_prevs = []
        new_att_kvs = []
        new_ffn_x_prevs = []

        for i, block in enumerate(self.model.blocks):
            layer_state = {
                'att_x_prev': att_x_prevs[i],
                'att_kv': att_kvs[i],
                'ffn_x_prev': ffn_x_prevs[i]
            }

            x, v_first, new_layer_state = block.forward_with_state(x, v_first, layer_state)

            new_att_x_prevs.append(new_layer_state['att_x_prev'])
            new_att_kvs.append(new_layer_state['att_kv'])
            new_ffn_x_prevs.append(new_layer_state['ffn_x_prev'])

        x = self.model.ln_out(x)

        return x, (
            torch.stack(new_att_x_prevs),
            torch.stack(new_att_kvs),
            torch.stack(new_ffn_x_prevs)
        )

    def _execute_decode(self, decode_batch: List[SequenceState]) -> List[Tuple[int, int]]:
        """
        Execute decode step.

        Returns:
            new_tokens: [(seq_id, token_id), ...]
        """
        if not decode_batch:
            return []

        if DEBUG_MEMORY:
            log_memory(f"Before decode batch (batch={len(decode_batch)})")

        indices = [s.slot_id for s in decode_batch]
        att_x, att_kv, ffn_x = self._gather_states(indices)

        input_tensor = torch.tensor(
            [[s.last_token_id] for s in decode_batch],
            dtype=torch.long, device=self.device
        )

        final_x, new_states = self._model_forward_with_state(
            input_tensor, att_x, att_kv, ffn_x
        )
        self._scatter_states(indices, new_states)

        if DEBUG_MEMORY:
            log_memory(f"After decode forward (batch={len(decode_batch)})")

        logits = self.model.head(final_x[:, -1, :]).to(torch.float32)
        new_tokens = []

        for i, seq in enumerate(decode_batch):
            token_id, _ = self.sampler.sample(
                logits[i],
                temperature=seq.temperature,
                top_p=seq.top_p,
                top_k=seq.top_k,
                logit_bias=seq.logit_bias
            )

            seq.last_token_id = token_id
            seq.generated_tokens.append(token_id)
            new_tokens.append((seq.seq_id, token_id))

            if seq.is_finished:
                self._finish_sequence(seq)

        return new_tokens

    def _execute_prefill(
        self,
        prefill_batch: List[Tuple[SequenceState, List[int]]]
    ) -> List[Tuple[int, int]]:
        """
        Execute prefill step, grouping by chunk length to avoid padding overhead.

        Returns:
            new_tokens: [(seq_id, token_id), ...] for sequences that completed prefill
        """
        if not prefill_batch:
            return []

        new_tokens = []

        jobs_by_len = collections.defaultdict(list)
        for seq, chunk in prefill_batch:
            jobs_by_len[len(chunk)].append((seq, chunk))

        for length, jobs in jobs_by_len.items():
            indices = [j[0].slot_id for j in jobs]
            chunks = [j[1] for j in jobs]

            if DEBUG_MEMORY:
                log_memory(f"Before prefill batch (len={length}, batch={len(jobs)})")

            att_x, att_kv, ffn_x = self._gather_states(indices)
            input_tensor = torch.tensor(chunks, dtype=torch.long, device=self.device)

            if DEBUG_MEMORY:
                print(f"[DEBUG] Prefill: input_tensor.shape={input_tensor.shape}, "
                      f"att_x.shape={att_x.shape}, att_kv.shape={att_kv.shape}")

            final_x, new_states = self._model_forward_with_state(
                input_tensor, att_x, att_kv, ffn_x
            )

            if DEBUG_MEMORY:
                log_memory(f"After prefill forward (len={length})")

            self._scatter_states(indices, new_states)

            for i, (seq, chunk) in enumerate(jobs):
                seq.prefill_offset += len(chunk)

                if seq.prefill_offset >= len(seq.prompt_tokens):
                    seq.status = SeqStatus.RUNNING
                    self.prefilling_seqs.remove(seq)
                    self.running_seqs.append(seq)

                    if self.state_cache is not None and seq.prompt_tokens:
                        state_backup = self._back_state_from_slot(seq.slot_id)
                        self.state_cache.create_checkpoints(
                            tokens=seq.prompt_tokens,
                            messages=[],
                            state=state_backup,
                            output=final_x[i, -1, :].cpu(),
                        )

                    logits = self.model.head(final_x[i, -1, :]).to(torch.float32)
                    token_id, _ = self.sampler.sample(
                        logits,
                        temperature=seq.temperature,
                        top_p=seq.top_p,
                        top_k=seq.top_k,
                        logit_bias=seq.logit_bias
                    )

                    seq.last_token_id = token_id
                    seq.generated_tokens.append(token_id)
                    new_tokens.append((seq.seq_id, token_id))

                    if seq.is_finished:
                        self._finish_sequence(seq)

            del final_x, new_states, att_x, att_kv, ffn_x, input_tensor

        return new_tokens

    # =========================================================================
    # Synchronous Execution Interface (used by BatchGenerator)
    # =========================================================================

    def step_sync(self) -> List[Tuple[int, int]]:
        """
        Execute one synchronous step.

        Returns:
            new_tokens: [(seq_id, token_id), ...]
        """
        max_attempts = 10

        for attempt in range(max_attempts):
            try:
                with torch.no_grad():
                    decode_batch, prefill_batch = self._schedule()

                    if not decode_batch and not prefill_batch:
                        if not self.has_pending_work():
                            return []
                        if attempt < 5:
                            time.sleep(0.01)
                            continue
                        else:
                            return []

                    new_tokens = []

                    if decode_batch:
                        decode_result = self._execute_decode(decode_batch)
                        new_tokens.extend(decode_result)

                    if prefill_batch:
                        new_tokens.extend(self._execute_prefill(prefill_batch))

                    if hasattr(self, '_step_counter'):
                        self._step_counter += 1
                    else:
                        self._step_counter = 1

                    if self._step_counter % 10 == 0:
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()

                    return new_tokens

            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e):
                    print(f"[ERROR] step_sync attempt {attempt} failed with CUDA error: {e}")
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(0.5)
                else:
                    raise

        return []

    def run_to_completion_sync(self) -> Iterator[List[Tuple[int, int]]]:
        """
        Run synchronously until all sequences complete.

        Yields:
            New tokens per step: [(seq_id, token_id), ...]
        """
        start_time = time.time()
        step_count = 0
        last_progress_step = 0
        total_generated_tokens = 0

        while self.has_pending_work():
            step_start = time.time()
            new_tokens = self.step_sync()

            if new_tokens:
                last_progress_step = step_count
                total_generated_tokens += len(new_tokens)

            step_time = time.time() - step_start
            if step_time > 5.0:
                stats = self.get_stats()
                print(f"[WARNING] Step {step_count} took {step_time:.2f}s! "
                      f"waiting={stats['waiting']}, prefilling={stats['prefilling']}, running={stats['running']}")

            if new_tokens:
                yield new_tokens
            step_count += 1

            if step_count > 100000:
                print(f"[ERROR] Too many steps ({step_count}), breaking to prevent infinite loop")
                final_stats = self.get_stats()
                print(f"[ERROR] Final stats: waiting={final_stats['waiting']}, prefilling={final_stats['prefilling']}, running={final_stats['running']}")
                break

            if step_count % 100 == 0:
                stats = self.get_stats()
                elapsed = time.time() - start_time
                print(f"[Scheduler] Step {step_count}, elapsed={elapsed:.1f}s, "
                      f"waiting={stats['waiting']}, prefilling={stats['prefilling']}, "
                      f"running={stats['running']}, total_tokens={total_generated_tokens}")

                steps_since_progress = step_count - last_progress_step
                if steps_since_progress > 500 and elapsed > 60:
                    print(f"[ERROR] No progress for {steps_since_progress} steps (>60s), breaking to prevent hang")
                    break

        total_time = time.time() - start_time
        print(f"[Scheduler] Completed in {total_time:.2f}s, total_steps={step_count}")

    # =========================================================================
    # Async Execution Interface (used by RWKVEngine)
    # =========================================================================

    async def step_async(self) -> List[Tuple[Union[int, str], int]]:
        """
        Execute one async step. The actual forward pass is synchronous (GPU);
        async is used for I/O operations like pushing to output queues.

        Returns:
            new_tokens: [(seq_id, token_id), ...]
        """
        return self.step_sync()


# =========================================================================
# Factory Function
# =========================================================================

def create_scheduler_core(
    model,
    tokenizer,
    max_batch_size: int = 64,
    max_batch_tokens: int = 8192,
    prefill_chunk_size: int = 512,
    state_cache: Optional["StateCache"] = None,
) -> SchedulerCore:
    """
    Create a SchedulerCore instance.

    Args:
        model: RWKV model
        tokenizer: Tokenizer
        max_batch_size: Maximum concurrent sequences
        max_batch_tokens: Maximum tokens per forward pass
        prefill_chunk_size: Chunked prefill chunk size
        state_cache: Optional State Cache instance
    """
    return SchedulerCore(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=max_batch_size,
        max_batch_tokens=max_batch_tokens,
        prefill_chunk_size=prefill_chunk_size,
        state_cache=state_cache,
    )
