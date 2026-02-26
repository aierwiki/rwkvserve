"""
Model configuration loader.

Supports multiple configuration methods:
1. Short config name: "0.1b" -> "rwkv7-0.1b.json"
2. Full config name: "rwkv7-0.1b" -> "rwkv7-0.1b.json"
3. Config file path: "/path/to/config.json"
"""
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ModelConfig:
    """Model configuration class."""

    def __init__(self, config_dict: Dict[str, Any], config_source: str = ""):
        self.config_dict = config_dict
        self.config_source = config_source

        self.architecture_version = config_dict.get("architecture_version", "rwkv7")

        self.model_name = config_dict.get("model_name", "Unknown")
        self.description = config_dict.get("description", "")

        self.model_file = config_dict.get("model_file", None)

        arch = config_dict.get("architecture", {})
        self.n_layer = arch.get("n_layer", 12)
        self.n_embd = arch.get("n_embd", 768)
        self.vocab_size = arch.get("vocab_size", 65536)
        self.head_size_a = arch.get("head_size_a", 64)
        self.ctx_len = arch.get("ctx_len", 4096)

        lora = config_dict.get("lora_dims", {})
        self.dim_att_lora = lora.get("dim_att_lora", 64)
        self.dim_gate_lora = lora.get("dim_gate_lora", 128)
        self.dim_mv_lora = lora.get("dim_mv_lora", 32)

        training = config_dict.get("training", {})
        self.recommended_ctx_len = training.get("recommended_ctx_len", 1024)
        self.recommended_batch_size = training.get("recommended_batch_size", 4)
        self.recommended_lr = training.get("recommended_lr", 1e-4)

    @property
    def model_path(self) -> Optional[str]:
        """Return the model file path, or None if not specified."""
        return self.model_file

    def to_train_config_dict(self) -> Dict[str, Any]:
        """Convert to training configuration dictionary."""
        return {
            'n_layer': self.n_layer,
            'n_embd': self.n_embd,
            'vocab_size': self.vocab_size,
            'head_size_a': self.head_size_a,
            'dim_att_lora': self.dim_att_lora,
            'dim_gate_lora': self.dim_gate_lora,
            'dim_mv_lora': self.dim_mv_lora,
        }

    def __repr__(self):
        return (f"ModelConfig({self.model_name}, "
                f"arch={self.architecture_version}, "
                f"n_layer={self.n_layer}, n_embd={self.n_embd})")


def list_available_models(config_dir: Optional[str] = None) -> Dict[str, str]:
    """
    List all available model configurations.

    Returns:
        Dictionary mapping model short names to config file paths.
    """
    if config_dir is None:
        config_dir = os.path.join(os.path.dirname(__file__), 'models')

    models = {}
    if not os.path.exists(config_dir):
        return models

    for filename in os.listdir(config_dir):
        if filename.endswith('.json'):
            base_name = filename.replace('.json', '')

            if '-' in base_name:
                parts = base_name.split('-', 1)
                if len(parts) == 2:
                    arch_version, short_name = parts
                    models[short_name] = os.path.join(config_dir, filename)
                    models[base_name] = os.path.join(config_dir, filename)
            else:
                models[base_name] = os.path.join(config_dir, filename)

    return models


def load_model_config(config_identifier: str,
                     config_dir: Optional[str] = None,
                     default_arch: str = "rwkv7") -> ModelConfig:
    """
    Load a model configuration (supports multiple methods).

    Args:
        config_identifier: Configuration identifier, supports three formats:
            1. Short name: "0.1b", "1.5b" -> auto-finds "{default_arch}-{size}.json"
            2. Full name: "rwkv7-0.1b" -> finds "rwkv7-0.1b.json"
            3. File path: "/path/to/config.json" or "./custom.json"
        config_dir: Configuration file directory, defaults to rwkvserve/configs/models
        default_arch: Default architecture version for short names

    Returns:
        ModelConfig object.
    """
    if config_dir is None:
        config_dir = os.path.join(os.path.dirname(__file__), 'models')

    config_file = None
    config_source = ""

    if '/' in config_identifier or config_identifier.endswith('.json'):
        config_file = config_identifier
        if not os.path.isabs(config_file):
            config_file = os.path.abspath(config_file)
        config_source = f"file:{config_file}"

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")

    else:
        available = list_available_models(config_dir)

        if config_identifier in available:
            config_file = available[config_identifier]
            config_source = f"builtin:{config_identifier}"
        else:
            full_name = f"{default_arch}-{config_identifier}"
            if full_name in available:
                config_file = available[full_name]
                config_source = f"builtin:{full_name}"
            else:
                raise FileNotFoundError(
                    f"Config not found: '{config_identifier}'\n"
                    f"Available configs: {sorted(set(available.keys()))}\n"
                    f"Or provide a full config file path"
                )

    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    return ModelConfig(config_dict, config_source=config_source)


def print_model_info(config: ModelConfig):
    """Print model configuration info."""
    print("=" * 80)
    print(f"Model Config: {config.model_name}")
    print("=" * 80)
    print(f"Architecture: {config.architecture_version}")
    print(f"Source: {config.config_source}")
    print(f"Description: {config.description}")
    if config.model_file:
        print(f"Model file: {config.model_file}")
    else:
        print(f"Model file: (not specified, use --load_model)")
    print()
    print("Architecture parameters:")
    print(f"  n_layer: {config.n_layer}")
    print(f"  n_embd: {config.n_embd}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  head_size: {config.head_size_a}")
    print(f"  n_head: {config.n_embd // config.head_size_a}")
    print()
    print("LORA dimensions:")
    print(f"  dim_att_lora: {config.dim_att_lora}")
    print(f"  dim_gate_lora: {config.dim_gate_lora}")
    print(f"  dim_mv_lora: {config.dim_mv_lora}")
    print()
    print("Recommended training config:")
    print(f"  ctx_len: {config.recommended_ctx_len}")
    print(f"  batch_size: {config.recommended_batch_size}")
    print(f"  learning_rate: {config.recommended_lr}")
    print("=" * 80)


if __name__ == "__main__":
    print("\nAvailable model configurations:")
    print("-" * 80)
    available = list_available_models()
    seen = set()
    for key, path in sorted(available.items()):
        if path not in seen:
            seen.add(path)
            try:
                config = load_model_config(key)
                print(f"  {key:15s} -> {config.model_name:20s} "
                      f"({config.architecture_version}, {config.n_layer} layers, {config.n_embd} dim)")
            except Exception as e:
                print(f"  {key:15s} -> Load failed: {e}")

    print("\n" + "=" * 80)
    print("Config loading test:")
    print("-" * 80)

    print("\nMethod 1 - Short name: load_model_config('1.5b')")
    config1 = load_model_config("1.5b")
    print(f"  Result: {config1}")

    print("\nMethod 2 - Full name: load_model_config('rwkv7-0.1b')")
    config2 = load_model_config("rwkv7-0.1b")
    print(f"  Result: {config2}")

    print("\n" + "=" * 80)
    print("Detailed config example:")
    print_model_info(config1)
