"""
RWKV Batch Inference Engine

A thin wrapper around SchedulerCore providing a convenient batch generation interface,
suitable for GRPO training and other batch inference scenarios.
"""

import torch
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from rwkvserve.inference.scheduler_core import (
    SequenceState, 
    SeqStatus,
    create_scheduler_core,
)


ReqStatus = SeqStatus


@dataclass
class GenerationConfig:
    """Generation configuration"""
    max_length: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    num_generations: int = 1
    
    max_batch_tokens: int = 8192
    chunk_size: int = 512


class BatchGenerator:
    """
    Batch inference engine - synchronous wrapper around SchedulerCore.
    
    Features:
    - Continuous Batching: parallel generation of multiple sequences
    - Chunked Prefill: controls peak memory for long prompts
    - State Cache: accelerates repeated inference with shared prefixes (optional)
    
    Usage:
        generator = BatchGenerator(model, tokenizer, use_state_cache=True)
        results = generator.generate(prompts, generation_config)
    """
    
    def __init__(self, model, tokenizer, config=None, use_state_cache: bool = False, **kwargs):
        """
        Args:
            model: RWKV model
            tokenizer: Tokenizer
            config: Optional configuration (kept for compatibility)
            use_state_cache: Whether to enable State Cache
            **kwargs: Ignored (kept for compatibility)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        self.eos_id = getattr(tokenizer, 'eos_token_id', None)
        
        self.state_cache = None
        if use_state_cache:
            from rwkvserve.inference.state_cache import create_state_cache
            self.state_cache = create_state_cache(
                model=model,
                cache_level="prefix",
            )
            print(f"BatchGenerator: State Cache enabled")
        
        # Lazily created on first generate() call to avoid repeated State Pool allocation
        self._scheduler = None
    
    def generate(
        self,
        prompts: Optional[List[str]] = None,
        generation_config: GenerationConfig = None,
        input_ids: Optional[List[List[int]]] = None,
    ) -> Dict[str, Any]:
        """
        Batch generation using Continuous Batching.
        
        Args:
            prompts: List of prompt strings [B]
            generation_config: Generation configuration
            input_ids: Pre-tokenized inputs [B] (takes priority over prompts)
            
        Returns:
            Dict with completions [B*G], completion_ids, masks
        """
        if input_ids is None:
            if prompts is None:
                raise ValueError("Must provide prompts or input_ids")
            input_ids = [self.tokenizer.encode(p) for p in prompts]
        
        B = len(input_ids)
        G = generation_config.num_generations
        total = B * G
        
        expanded_ids = [list(ids) for ids in input_ids for _ in range(G)]
        
        max_batch_size = min(getattr(self.config, "generation_max_batch_size", 256), total)
        
        if self._scheduler is None or \
           self._scheduler.max_batch_size < total or \
           self._scheduler.max_batch_tokens != generation_config.max_batch_tokens or \
           self._scheduler.prefill_chunk_size != generation_config.chunk_size:
            self._scheduler = create_scheduler_core(
                model=self.model,
                tokenizer=self.tokenizer,
                max_batch_size=max_batch_size,
                max_batch_tokens=generation_config.max_batch_tokens,
                prefill_chunk_size=generation_config.chunk_size,
                state_cache=self.state_cache,
            )
        else:
            self._scheduler.reset()
        
        scheduler = self._scheduler
        
        sequences = [
            SequenceState(
                seq_id=i,
                prompt_tokens=tokens,
                max_new_tokens=generation_config.max_length,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
                eos_token_id=self.eos_id,
            )
            for i, tokens in enumerate(expanded_ids)
        ]
        scheduler.add_sequences(sequences)
        
        import time
        
        all_tokens = [[] for _ in range(total)]
        
        gen_start = time.time()
        for updates in scheduler.run_to_completion_sync():
            for seq_id, token_id in updates:
                all_tokens[seq_id].append(token_id)
        
        gen_time = time.time() - gen_start
        avg_len = sum(len(t) for t in all_tokens) / total if total > 0 else 0
        print(f"[BatchGenerator] Generation complete: {total} sequences, "
              f"avg_len={avg_len:.1f}, time={gen_time:.2f}s, "
              f"throughput={total*avg_len/gen_time:.1f} tokens/s")
        
        return self._postprocess(all_tokens, total, generation_config)
    
    def _postprocess(self, all_tokens: List[List[int]], total: int, generation_config: GenerationConfig) -> Dict[str, Any]:
        """Post-process: decode text and convert to tensors."""
        completions = []
        for i, tokens in enumerate(all_tokens):
            if self.eos_id is not None and tokens and tokens[-1] == self.eos_id:
                tokens = tokens[:-1]
                all_tokens[i] = tokens
            completions.append(self.tokenizer.decode(tokens))
        
        max_len = max((len(t) for t in all_tokens), default=0)
        
        if max_len > 0:
            padded_ids = []
            masks = []
            for tokens in all_tokens:
                pad_len = max_len - len(tokens)
                padded_ids.append(tokens + [0] * pad_len)
                masks.append([1] * len(tokens) + [0] * pad_len)
            
            completion_ids = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
            masks_tensor = torch.tensor(masks, dtype=torch.float32, device=self.device)
        else:
            completion_ids = torch.zeros((total, 0), dtype=torch.long, device=self.device)
            masks_tensor = torch.zeros((total, 0), dtype=torch.float32, device=self.device)
        
        return {
            'completions': completions,
            'completion_ids': completion_ids,
            'completion_logps': torch.tensor([], device=self.device),
            'masks': masks_tensor,
        }
