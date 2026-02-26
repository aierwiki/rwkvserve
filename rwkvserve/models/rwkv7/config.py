"""
Simplified configuration for RWKV7 models.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SimpleTrainConfig:
    """RWKV7 model configuration supporting both training and inference."""

    load_model: str = ""
    n_layer: int = 12
    n_embd: int = 768
    vocab_size: int = 65536

    ctx_len: int = 1024
    micro_bsz: int = 4
    epoch_count: int = 10
    epoch_steps: int = 1000
    epoch_save: int = 5
    save_every_n_batches: int = 0
    epoch_begin: int = 0
    accumulate_grad_batches: int = 1
    grad_cp: int = 0

    data_file: str = ""
    sft_field: List[str] = field(default_factory=lambda: ["instruction", "output"])
    sft_split: str = "train"
    num_workers: int = 0
    chunk_size: int = 2048
    cache_dir: Optional[str] = None

    lr_init: float = 3e-4
    lr_final: float = 1e-5
    warmup_steps: int = 50
    beta1: float = 0.9
    beta2: float = 0.99
    adam_eps: float = 1e-18
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    layerwise_lr: int = 1

    accelerator: str = "gpu"
    devices: int = 1
    strategy: str = "auto"
    precision: str = "bf16"

    proj_dir: str = "output"
    seed: int = 42
    head_size_a: int = 64
    report_to: str = ""
    run_name: str = ""

    train_type: str = "normal"
    chunk_ctx: int = 512

    dim_att_lora: int = 0
    dim_gate_lora: int = 0
    dim_mv_lora: int = 0

    def __post_init__(self):
        self.dim_att = self.n_embd
        self.dim_ffn = int((self.n_embd * 3.5) // 32 * 32)
        self.head_size_divisor = 8
        self.my_testing = 'x070'

        if self.train_type == "infctx":
            assert self.chunk_ctx > 0, "chunk_ctx must be > 0 for infctx training"
            assert self.chunk_ctx <= self.ctx_len, "chunk_ctx must be <= ctx_len"
            assert self.ctx_len % self.chunk_ctx == 0, "ctx_len must be divisible by chunk_ctx"
