"""
RWKVServe: RWKV Inference & Serving

Provides:
- OpenAI-compatible API server with Continuous Batching
- vLLM-style batch inference API (LLM + SamplingParams)
- State Cache for accelerated inference
- Multi-model and data parallel support
"""

__version__ = "0.1.0"
__author__ = "RWKVServe Contributors"
__license__ = "Apache-2.0"

from rwkvserve.models import (
    RWKV7Model,
    RWKV7Config,
    RWKVConfig,
    GenerationConfig,
    AutoModel,
)
from rwkvserve.inference import InferencePipeline, TextGenerator
from rwkvserve.data.tokenizers import (
    get_tokenizer,
    TRIE_TOKENIZER,
    AutoTokenizer,
    RWKVTokenizer,
)
from rwkvserve.utils import create_model_hub
from rwkvserve.entrypoints import LLM
from rwkvserve.sampling_params import SamplingParams
from rwkvserve.outputs import RequestOutput, CompletionOutput
from rwkvserve.peft import load_and_merge_lora

__all__ = [
    # Models
    "RWKV7Model",
    "RWKV7Config",
    "RWKVConfig",
    "GenerationConfig",
    "AutoModel",
    # Inference
    "InferencePipeline",
    "TextGenerator",
    # vLLM-style batch inference
    "LLM",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    # Tokenizer
    "get_tokenizer",
    "TRIE_TOKENIZER",
    "AutoTokenizer",
    "RWKVTokenizer",
    # LoRA / PEFT
    "load_and_merge_lora",
    # Utilities
    "create_model_hub",
    # Metadata
    "__version__",
]
