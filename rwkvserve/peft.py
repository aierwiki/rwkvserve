"""
LoRA adapter loading and merging for inference serving.

Follows vLLM's LoRA serving pattern:
- Server registers LoRA adapters at startup via --lora-modules name=path
- Each adapter is merged into the base model weights for zero-overhead inference
- API requests select the adapter via the `model` field

Supports adapter formats produced by rwkvtune and HuggingFace PEFT:
- adapter_config.json / lora_config.json  (config)
- adapter_model.bin / adapter_model.safetensors / lora_weights.pth  (weights)
"""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


@dataclass
class LoRAModuleConfig:
    """A registered LoRA module (mirrors vLLM's --lora-modules format)."""
    name: str
    path: str
    base_model_name: Optional[str] = None

    @classmethod
    def from_string(cls, spec: str) -> "LoRAModuleConfig":
        """Parse 'name=path' or JSON format, matching vLLM CLI conventions."""
        spec = spec.strip()
        if spec.startswith("{"):
            d = json.loads(spec)
            return cls(
                name=d["name"],
                path=d["path"],
                base_model_name=d.get("base_model_name"),
            )
        if "=" not in spec:
            raise ValueError(
                f"Invalid --lora-modules format: {spec!r}\n"
                f"Expected: name=path  (e.g. my-lora=/path/to/adapter)"
            )
        name, path = spec.split("=", 1)
        return cls(name=name.strip(), path=path.strip())


@dataclass
class LoRAAdapterConfig:
    """Parsed LoRA adapter config (from adapter_config.json)."""
    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    target_modules: Optional[List[str]] = None
    modules_to_save: Optional[List[str]] = None
    use_rslora: bool = False

    @property
    def scaling(self) -> float:
        if self.use_rslora:
            return self.lora_alpha / math.sqrt(self.r)
        return self.lora_alpha / self.r

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "LoRAAdapterConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            r=d.get("r", 64),
            lora_alpha=d.get("lora_alpha", 128),
            lora_dropout=d.get("lora_dropout", 0.0),
            target_modules=d.get("target_modules"),
            modules_to_save=d.get("modules_to_save"),
            use_rslora=d.get("use_rslora", False),
        )


def _find_config_file(adapter_dir: Path) -> Path:
    for name in ("adapter_config.json", "lora_config.json"):
        p = adapter_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"LoRA config not found in {adapter_dir}. "
        f"Expected adapter_config.json or lora_config.json"
    )


def _find_weights_file(adapter_dir: Path) -> Path:
    for name in ("adapter_model.bin", "adapter_model.safetensors", "lora_weights.pth"):
        p = adapter_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"LoRA weights not found in {adapter_dir}. "
        f"Expected adapter_model.bin, adapter_model.safetensors, or lora_weights.pth"
    )


def _load_weights(weights_path: Path) -> Dict[str, torch.Tensor]:
    suffix = weights_path.suffix
    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
            return load_file(str(weights_path))
        except ImportError:
            raise ImportError("safetensors required: pip install safetensors")
    state = torch.load(str(weights_path), map_location="cpu")
    if isinstance(state, dict) and "lora_state_dict" in state:
        return state["lora_state_dict"]
    return state


def _is_target_module(name: str, targets: List[str]) -> bool:
    """Check if a module name matches any of the target patterns."""
    for target in targets:
        if target in name:
            return True
    return False


def load_and_merge_lora(
    model: nn.Module,
    adapter_path: Union[str, Path],
) -> Tuple[nn.Module, LoRAAdapterConfig]:
    """
    Load a LoRA adapter and merge its weights into the base model.

    This is the recommended approach for inference serving: merging LoRA
    weights eliminates runtime overhead (no extra matmuls). The merged
    model behaves identically to a fully fine-tuned checkpoint.

    Args:
        model: Base model (nn.Module) with weights already loaded.
        adapter_path: Directory containing adapter config and weights.

    Returns:
        Tuple of (model with merged weights, adapter config).
    """
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"LoRA adapter path does not exist: {adapter_path}")

    config = LoRAAdapterConfig.from_json(_find_config_file(adapter_path))
    lora_state = _load_weights(_find_weights_file(adapter_path))
    scaling = config.scaling

    print(f"  LoRA config: r={config.r}, alpha={config.lora_alpha}, "
          f"scaling={scaling:.4f}, targets={config.target_modules}")

    target_modules = config.target_modules or []

    # Build a map: module_name -> nn.Linear for fast lookup
    linear_modules: Dict[str, nn.Linear] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _is_target_module(name, target_modules):
            linear_modules[name] = module

    merged_count = 0
    saved_count = 0

    # Strategy 1: key format "module_name.lora_A" / "module_name.lora_B"
    # (produced by rwkvtune PEFT save)
    lora_a_keys = {k for k in lora_state if k.endswith(".lora_A")}
    lora_b_keys = {k for k in lora_state if k.endswith(".lora_B")}

    paired_modules = set()
    for a_key in lora_a_keys:
        module_name = a_key[: -len(".lora_A")]
        b_key = f"{module_name}.lora_B"
        if b_key not in lora_b_keys:
            continue
        paired_modules.add(module_name)

        if module_name in linear_modules:
            linear = linear_modules[module_name]
            lora_a = lora_state[a_key].to(dtype=linear.weight.dtype, device=linear.weight.device)
            lora_b = lora_state[b_key].to(dtype=linear.weight.dtype, device=linear.weight.device)
            # delta_W = B @ A * scaling  (out_features x in_features)
            with torch.no_grad():
                linear.weight.data += (lora_b @ lora_a) * scaling
            merged_count += 1

    # Strategy 2: keys that don't match lora_A/lora_B pattern
    # These are "modules_to_save" — full parameter overrides (e.g. LayerNorm, time-mix)
    used_keys = set()
    for a_key in lora_a_keys:
        used_keys.add(a_key)
    for b_key in lora_b_keys:
        used_keys.add(b_key)

    remaining_keys = set(lora_state.keys()) - used_keys
    if remaining_keys:
        model_state = model.state_dict()
        updates = {}
        for key in remaining_keys:
            if key in model_state:
                updates[key] = lora_state[key].to(
                    dtype=model_state[key].dtype,
                    device=model_state[key].device,
                )
                saved_count += 1
        if updates:
            model.load_state_dict(updates, strict=False)

    print(f"  Merged {merged_count} LoRA modules, "
          f"overrode {saved_count} saved modules")

    return model, config
