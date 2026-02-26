"""
RWKV Model Configuration.
Supports loading from and saving to config.json, following the Transformers pattern.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


@dataclass
class RWKVConfig:
    """
    Standard RWKV model configuration.

    Args:
        model_type: Model type, e.g. "rwkv7"
        architectures: Model architecture class names
        n_layer: Number of layers
        n_embd: Embedding dimension
        vocab_size: Vocabulary size
        head_size_a: Attention head size
        ctx_len: Context length
        dim_att_lora: Attention LORA dimension
        dim_gate_lora: Gate LORA dimension
        dim_mv_lora: MV LORA dimension
        bos_token_id: BOS token ID
        eos_token_id: EOS token ID
        pad_token_id: PAD token ID
    """

    model_type: str = "rwkv7"
    architectures: List[str] = field(default_factory=lambda: ["RWKV7Model"])

    n_layer: int = 12
    n_embd: int = 768
    vocab_size: int = 65536
    head_size_a: int = 64
    ctx_len: int = 4096

    dim_att_lora: int = 64
    dim_gate_lora: int = 128
    dim_mv_lora: int = 32

    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = 261
    pad_token_id: Optional[int] = 0

    dim_att: Optional[int] = None
    dim_ffn: Optional[int] = None
    head_size_divisor: int = 8

    my_testing: str = "x070"
    transformers_version: str = "rwkvserve-0.1.0"

    use_cache: bool = True
    tie_word_embeddings: bool = False

    _extra_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute derived parameters after init."""
        if self.dim_att is None:
            self.dim_att = self.n_embd

        if self.dim_ffn is None:
            self.dim_ffn = int((self.n_embd * 3.5) // 32 * 32)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RWKVConfig":
        """Create a config from a dictionary."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values() if f.name != '_extra_config'}
        init_kwargs = {}
        extra_config = {}

        for key, value in config_dict.items():
            if key in known_fields:
                init_kwargs[key] = value
            else:
                extra_config[key] = value

        if extra_config:
            init_kwargs['_extra_config'] = extra_config

        return cls(**init_kwargs)

    @classmethod
    def from_json_file(cls, json_file: str) -> "RWKVConfig":
        """Load config from a JSON file."""
        with open(json_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> "RWKVConfig":
        """Load config from a pretrained model directory."""
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, "config.json")
            if not os.path.exists(config_file):
                raise FileNotFoundError(
                    f"Config file not found: {config_file}\n"
                    f"Ensure the model directory contains a config.json file"
                )
            return cls.from_json_file(config_file)

        raise NotImplementedError(f"Remote loading not yet supported: {pretrained_model_name_or_path}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        config_dict = {}
        for key, value in asdict(self).items():
            if key == '_extra_config':
                config_dict.update(value)
            elif value is not None:
                config_dict[key] = value

        return config_dict

    def to_json_file(self, json_file: str, **kwargs):
        """Save config to a JSON file."""
        config_dict = self.to_dict()

        json_kwargs = {
            'indent': 2,
            'ensure_ascii': False,
        }
        json_kwargs.update(kwargs)

        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, **json_kwargs)

    def save_pretrained(self, save_directory: str):
        """Save config to a directory."""
        os.makedirs(save_directory, exist_ok=True)
        config_file = os.path.join(save_directory, "config.json")
        self.to_json_file(config_file)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} {json.dumps(self.to_dict(), indent=2, ensure_ascii=False)}"


@dataclass
class GenerationConfig:
    """Configuration for text generation parameters."""

    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = 261
    pad_token_id: Optional[int] = 0

    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.85
    top_k: int = 0
    repetition_penalty: float = 1.0

    max_length: int = 1024
    max_new_tokens: Optional[int] = None

    num_return_sequences: int = 1

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GenerationConfig":
        """Create generation config from a dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

    @classmethod
    def from_json_file(cls, json_file: str) -> "GenerationConfig":
        """Load generation config from a JSON file."""
        with open(json_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> "GenerationConfig":
        """Load generation config from a pretrained model directory."""
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, "generation_config.json")
            if os.path.exists(config_file):
                return cls.from_json_file(config_file)
            else:
                return cls()

        raise NotImplementedError(f"Remote loading not yet supported: {pretrained_model_name_or_path}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json_file(self, json_file: str, **kwargs):
        """Save to a JSON file."""
        config_dict = self.to_dict()
        json_kwargs = {'indent': 2, 'ensure_ascii': False}
        json_kwargs.update(kwargs)

        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, **json_kwargs)

    def save_pretrained(self, save_directory: str):
        """Save to a directory."""
        os.makedirs(save_directory, exist_ok=True)
        config_file = os.path.join(save_directory, "generation_config.json")
        self.to_json_file(config_file)
