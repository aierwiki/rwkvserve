"""
AutoModel - Automatic RWKV model loading.
Follows the Transformers AutoModel design pattern.
"""

import os
import torch
from typing import Optional, Union
from .configuration import RWKVConfig


MODEL_MAPPING = {
    "rwkv7": "RWKV7Model",
}


class AutoModel:
    """
    AutoModel - Automatically selects and loads the appropriate RWKV model.

    Selects the model class based on the model_type field in the config.

    Usage:
        model = AutoModel.from_pretrained("path/to/model")

        config = RWKVConfig.from_json_file("config.json")
        model = AutoModel.from_config(config)
        model.load_state_dict(torch.load("model.pth"))
    """

    @staticmethod
    def _get_model_class(model_type: str):
        """
        Get the model class for the given model_type.

        Args:
            model_type: Model type string (e.g. "rwkv7")

        Returns:
            The model class.
        """
        if model_type not in MODEL_MAPPING:
            raise ValueError(
                f"Unsupported model type: {model_type}\n"
                f"Supported model types: {list(MODEL_MAPPING.keys())}"
            )

        if model_type == "rwkv7":
            from rwkvserve.models.rwkv7 import RWKV7Model
            return RWKV7Model

        raise NotImplementedError(f"Model type {model_type} is not yet implemented")

    @classmethod
    def from_config(cls, config: RWKVConfig, **kwargs):
        """
        Create a model from config (without loading weights).

        Args:
            config: RWKV config object.
            **kwargs: Extra arguments passed to the model constructor.

        Returns:
            Model instance.
        """
        model_class = cls._get_model_class(config.model_type)
        return model_class(config, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: Optional[Union[str, torch.device]] = "cpu",
        dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs
    ):
        """
        Load a model from a path (directory or file).

        Args:
            model_path: Model path. Can be:
                - Directory containing config.json and weight files (model.pth, etc.)
                - A .pth weight file (config.json is looked up in the same directory)
            device: Device ("cpu", "cuda", "cuda:0", etc.)
            dtype: Data type ("fp32", "fp16", "bf16" or torch.dtype)
            **kwargs: Extra arguments:
                - strict: Whether to strictly match weights (default False)
                - load_weights: Whether to load weights (default True)

        Returns:
            Model instance with loaded weights.
        """
        if os.path.isdir(model_path):
            config_dir = model_path
            weight_file = None

            for filename in ["model.pth", "pytorch_model.pth", "model.safetensors"]:
                candidate = os.path.join(model_path, filename)
                if os.path.exists(candidate):
                    weight_file = candidate
                    break

        elif os.path.isfile(model_path):
            weight_file = model_path
            config_dir = os.path.dirname(model_path)

            if not config_dir:
                config_dir = "."

        else:
            raise ValueError(
                f"Path does not exist: {model_path}\n"
                f"model_path should be a model directory or weight file (.pth)"
            )

        config = RWKVConfig.from_pretrained(config_dir)
        model = cls.from_config(config)

        if dtype is not None:
            if isinstance(dtype, str):
                dtype_map = {
                    "float32": torch.float32,
                    "fp32": torch.float32,
                    "float16": torch.float16,
                    "fp16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "bf16": torch.bfloat16,
                }
                dtype = dtype_map.get(dtype.lower(), torch.float32)

            model = model.to(dtype=dtype)

        load_weights = kwargs.pop("load_weights", True)
        if load_weights:
            if weight_file is None:
                raise FileNotFoundError(
                    f"Weight file not found. Supported filenames:\n"
                    f"  - model.pth\n"
                    f"  - pytorch_model.pth\n"
                    f"  - model.safetensors\n"
                    f"In directory: {config_dir}"
                )

            strict = kwargs.pop("strict", False)

            if weight_file.endswith(".safetensors"):
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(weight_file)
                except ImportError:
                    raise ImportError(
                        "safetensors is required to load .safetensors weights:\n"
                        "  pip install safetensors"
                    )
            else:
                map_location = "cpu"
                state_dict = torch.load(weight_file, map_location=map_location)

            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]

            model.load_state_dict(state_dict, strict=strict)

        if device is not None:
            model = model.to(device)

        return model

    @classmethod
    def save_pretrained(
        cls,
        model,
        save_directory: str,
        save_format: str = "pth",
        **kwargs
    ):
        """
        Save a model to a directory.

        Args:
            model: The model to save.
            save_directory: Output directory.
            save_format: Save format ("pth" or "safetensors").
            **kwargs: Extra arguments.
        """
        os.makedirs(save_directory, exist_ok=True)

        if hasattr(model, "config"):
            model.config.save_pretrained(save_directory)

        if save_format == "safetensors":
            try:
                from safetensors.torch import save_file
                weight_file = os.path.join(save_directory, "model.safetensors")
                save_file(model.state_dict(), weight_file)
            except ImportError:
                raise ImportError(
                    "safetensors is required to save .safetensors format:\n"
                    "  pip install safetensors"
                )
        else:
            weight_file = os.path.join(save_directory, "model.pth")
            torch.save(model.state_dict(), weight_file)
