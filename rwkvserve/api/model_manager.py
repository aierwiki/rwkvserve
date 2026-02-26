"""
Model Manager - Multi-Model Deployment and Data Parallelism

Features:
- Manage loading and unloading of multiple models
- Route requests to the appropriate model based on the model field
- Support per-model device assignment
- Support configuration file-based model loading
- State Cache integration
- Data-parallel inference (multi-GPU)
"""

import os
import yaml
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from rwkvserve.api.protocol import ModelCard, ModelList, ErrorResponse

if TYPE_CHECKING:
    from rwkvserve.api.async_serving_chat import AsyncChatCompletionService
    from rwkvserve.api.async_serving_completion import AsyncCompletionService
    from rwkvserve.inference.state_cache import StateCache
    from rwkvserve.engine.data_parallel import DataParallelEngine


@dataclass
class LoRAModuleSpec:
    """A LoRA module registered via --lora-modules (vLLM-compatible)."""
    name: str
    path: str


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    path: str
    device: str = "auto"
    dtype: Optional[str] = None
    
    max_batch_size: int = 1
    prefill_chunk_size: int = 512
    description: str = ""
    
    gpu_ids: Optional[List[int]] = None

    enable_lora: bool = False
    lora_modules: Optional[List[LoRAModuleSpec]] = None
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def is_data_parallel(self) -> bool:
        return self.gpu_ids is not None and len(self.gpu_ids) > 1
    
    @classmethod
    def from_string(cls, config_str: str) -> "ModelConfig":
        """
        Parse model configuration from a string.

        Format: name:path[:device][:dtype]

        Examples:
            model1:/path/to/model1
            model2:/path/to/model2:cuda:0
            model3:/path/to/model3:cpu:bf16
        """
        parts = config_str.split(":")
        
        if len(parts) < 2:
            raise ValueError(
                f"Invalid model config: {config_str}\n"
                f"Expected format: name:path[:device][:dtype]\n"
                f"Example: model1:/path/to/model1"
            )
        
        name = parts[0]
        remaining = parts[1:]
        
        dtype = None
        device = "auto"
        
        if remaining and remaining[-1].lower() in ("fp32", "fp16", "bf16", "float32", "float16", "bfloat16"):
            dtype = remaining.pop()
            
        if remaining:
            last = remaining[-1]
            if last in ("cpu", "cuda", "auto"):
                device = remaining.pop()
            elif last.isdigit() and len(remaining) >= 2 and remaining[-2] == "cuda":
                device = f"{remaining[-2]}:{remaining[-1]}"
                remaining.pop()
                remaining.pop()
        
        path = ":".join(remaining)
        
        return cls(name=name, path=path, device=device, dtype=dtype)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Parse model configuration from a dictionary."""
        return cls(
            name=config_dict["name"],
            path=config_dict["path"],
            device=config_dict.get("device", "auto"),
            dtype=config_dict.get("dtype"),
            max_batch_size=config_dict.get("max_batch_size", 1),
            prefill_chunk_size=config_dict.get("prefill_chunk_size", 512),
            description=config_dict.get("description", ""),
            gpu_ids=config_dict.get("gpu_ids"),
        )


@dataclass
class LoadedModel:
    """A loaded model instance."""
    config: ModelConfig
    model: Any
    tokenizer: Any
    chat_service: Any
    completion_service: Any
    state_cache: Optional[Any] = None
    data_parallel_engine: Optional[Any] = None
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def device(self) -> str:
        return self.config.device
    
    @property
    def path(self) -> str:
        return self.config.path
    
    @property
    def is_data_parallel(self) -> bool:
        return self.data_parallel_engine is not None


class ModelManager:
    """
    Model manager for multi-model deployment.

    Responsibilities:
    - Load and manage multiple models
    - Route requests by model name
    - List available models
    - Manage State Cache
    """
    
    def __init__(self, cache_config: Optional[Dict[str, Any]] = None, reasoning_parser=None):
        self._models: Dict[str, LoadedModel] = {}
        self._default_model: Optional[str] = None
        self._cache_config = cache_config or {
            "max_cache_memory_gb": 4.0,
            "cache_level": "prefix",
            "cache_first_message": True,
        }
        self._reasoning_parser = reasoning_parser
    
    @property
    def models(self) -> Dict[str, LoadedModel]:
        return self._models
    
    @property
    def model_names(self) -> List[str]:
        return list(self._models.keys())
    
    @property
    def default_model(self) -> Optional[str]:
        return self._default_model
    
    def load_model(self, config: ModelConfig) -> LoadedModel:
        """Load a single model."""
        from rwkvserve.models import AutoModel
        from rwkvserve.data.tokenizers import AutoTokenizer
        from rwkvserve.api.async_serving_chat import AsyncChatCompletionService
        from rwkvserve.api.async_serving_completion import AsyncCompletionService
        from rwkvserve.engine.core import RWKVEngine
        from rwkvserve.inference.state_cache import create_state_cache
        
        print(f"\n{'='*60}")
        print(f"Loading model: {config.name}")
        print(f"{'='*60}")
        print(f"Path: {config.path}")
        
        if not os.path.exists(config.path):
            raise FileNotFoundError(f"Model path does not exist: {config.path}")
        
        if config.is_data_parallel:
            return self._load_model_data_parallel(config)
        
        print(f"Device: {config.device}")
        if config.dtype:
            print(f"Precision: {config.dtype}")
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.path)
        print(f"Tokenizer loaded (vocab size: {len(tokenizer)})")
        
        print("Loading model...")
        model = AutoModel.from_pretrained(config.path, device=config.device, dtype=config.dtype)

        if config.enable_lora and config.lora_modules:
            from rwkvserve.peft import load_and_merge_lora
            for lora_spec in config.lora_modules:
                print(f"Loading LoRA adapter: {lora_spec.name}")
                print(f"  Path: {lora_spec.path}")
                model, _ = load_and_merge_lora(model, lora_spec.path)
                print(f"  LoRA adapter '{lora_spec.name}' merged into base model")

        model.eval()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model loaded (parameters: {param_count:,})")
        
        state_cache = None
        if self._cache_config.get("cache_level", "prefix") != "none":
            print("Initializing State Cache...")
            try:
                state_cache = create_state_cache(
                    model=model,
                    max_cache_memory_gb=self._cache_config.get("max_cache_memory_gb", 4.0),
                    cache_level=self._cache_config.get("cache_level", "prefix"),
                    cache_first_message=self._cache_config.get("cache_first_message", True),
                )
                print("State Cache initialized")
            except Exception as e:
                print(f"State Cache init failed: {e}, continuing without cache")
                state_cache = None
        
        print(f"Continuous Batching Engine enabled (Max Batch: {config.max_batch_size})")
        print(f"Chunked Prefill: {config.prefill_chunk_size} tokens/chunk")
        
        engine = RWKVEngine(
            model, 
            tokenizer, 
            max_batch_size=config.max_batch_size,
            prefill_chunk_size=config.prefill_chunk_size,
            state_cache=state_cache,
        )
        
        chat_service = AsyncChatCompletionService(
            engine=engine,
            model=model,
            tokenizer=tokenizer,
            model_name=config.name,
            state_cache=state_cache,
            reasoning_parser=self._reasoning_parser,
        )
        
        completion_service = AsyncCompletionService(
            engine=engine,
            model=model,
            tokenizer=tokenizer,
            model_name=config.name,
            state_cache=state_cache,
        )
        
        loaded_model = LoadedModel(
            config=config,
            model=model,
            tokenizer=tokenizer,
            chat_service=chat_service,
            completion_service=completion_service,
            state_cache=state_cache,
        )
        
        self._models[config.name] = loaded_model
        
        if config.enable_lora and config.lora_modules:
            for lora_spec in config.lora_modules:
                self._models[lora_spec.name] = loaded_model
                print(f"Registered LoRA alias: '{lora_spec.name}' -> '{config.name}'")
        
        if self._default_model is None:
            if config.enable_lora and config.lora_modules:
                self._default_model = config.lora_modules[0].name
            else:
                self._default_model = config.name
            print(f"Set as default model: {self._default_model}")
        
        print(f"{'='*60}\n")
        
        return loaded_model
    
    def _load_model_data_parallel(self, config: ModelConfig) -> LoadedModel:
        """Load a model in data-parallel mode across multiple GPUs."""
        from rwkvserve.data.tokenizers import AutoTokenizer
        from rwkvserve.engine.data_parallel import DataParallelEngine
        
        gpu_ids = config.gpu_ids
        print(f"Data-parallel mode: {len(gpu_ids)} GPUs")
        print(f"  GPU IDs: {gpu_ids}")
        if config.dtype:
            print(f"Precision: {config.dtype}")
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.path)
        print(f"Tokenizer loaded (vocab size: {len(tokenizer)})")
        
        dp_engine = DataParallelEngine(
            model_path=config.path,
            model_name=config.name,
            gpu_ids=gpu_ids,
            dtype=config.dtype,
            max_batch_size=config.max_batch_size,
            prefill_chunk_size=config.prefill_chunk_size,
            cache_config=self._cache_config,
        )
        
        from rwkvserve.api.dp_serving_chat import DataParallelChatService
        chat_service = DataParallelChatService(
            engine=dp_engine,
            tokenizer=tokenizer,
            model_name=config.name,
        )
        
        completion_service = None
        
        loaded_model = LoadedModel(
            config=config,
            model=None,
            tokenizer=tokenizer,
            chat_service=chat_service,
            completion_service=completion_service,
            state_cache=None,
            data_parallel_engine=dp_engine,
        )
        
        self._models[config.name] = loaded_model
        
        if self._default_model is None:
            self._default_model = config.name
            print("Set as default model")
        
        print(f"{'='*60}\n")
        
        return loaded_model
    
    def load_models(self, configs: List[ModelConfig]) -> List[LoadedModel]:
        """Load multiple models."""
        loaded = []
        for config in configs:
            try:
                model = self.load_model(config)
                loaded.append(model)
            except Exception as e:
                print(f"Failed to load model {config.name}: {e}")
                raise
        
        return loaded
    
    def load_from_yaml(self, yaml_path: str) -> List[LoadedModel]:
        """
        Load models from a YAML configuration file.

        Config format:
        ```yaml
        models:
          - name: model1
            path: /path/to/model1
            device: cuda:0
          - name: model2
            path: /path/to/model2
            device: cuda:1

        default_model: model1  # optional
        ```
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        
        models_data = config_data.get("models", [])
        if not models_data:
            raise ValueError(f"No model configurations found in: {yaml_path}")
        
        configs = [ModelConfig.from_dict(m) for m in models_data]
        
        loaded = self.load_models(configs)
        
        default_model = config_data.get("default_model")
        if default_model and default_model in self._models:
            self._default_model = default_model
            print(f"Default model set to: {default_model}")
        
        return loaded
    
    def get_loaded_model(self, name: Optional[str] = None) -> Optional[LoadedModel]:
        """Get a loaded model instance by name, or the default model if name is None."""
        if name is None:
            name = self._default_model
        return self._models.get(name)
    
    def get_model(self, name: Optional[str] = None) -> Optional[LoadedModel]:
        """Get a model (alias for get_loaded_model)."""
        if name is None:
            name = self._default_model
        
        if name is None:
            return None
        
        return self._models.get(name)
    
    def get_chat_service(self, model_name: Optional[str] = None) -> Optional[Any]:
        """Get the chat service for the specified model."""
        model = self.get_model(model_name)
        if model:
            return model.chat_service
        return None
    
    def get_completion_service(self, model_name: Optional[str] = None) -> Optional[Any]:
        """Get the completion service for the specified model."""
        model = self.get_model(model_name)
        if model:
            return model.completion_service
        return None
    
    def list_models(self) -> ModelList:
        """List all available models."""
        cards = []
        for name, loaded_model in self._models.items():
            card = ModelCard(
                id=name,
                owned_by="rwkvserve",
            )
            cards.append(card)
        
        return ModelList(data=cards)
    
    def model_exists(self, name: str) -> bool:
        return name in self._models
    
    def create_model_not_found_error(self, model_name: str) -> ErrorResponse:
        available = ", ".join(self._models.keys()) if self._models else "none"
        return ErrorResponse(
            message=f"Model '{model_name}' not found. Available models: {available}",
            type="InvalidRequestError",
            param="model",
            code="model_not_found",
        )
    
    def unload_model(self, name: str) -> bool:
        """Unload a model by name."""
        if name not in self._models:
            return False
        
        del self._models[name]
        
        if self._default_model == name:
            self._default_model = next(iter(self._models.keys()), None)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Model {name} unloaded")
        return True
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics for all models."""
        all_stats = {}
        for name, loaded_model in self._models.items():
            if loaded_model.state_cache is not None:
                all_stats[name] = loaded_model.state_cache.get_stats()
        
        if not all_stats:
            return None
        
        if len(all_stats) == 1:
            return list(all_stats.values())[0]
        
        return all_stats
    
    def __len__(self) -> int:
        return len(self._models)
    
    def __contains__(self, name: str) -> bool:
        return name in self._models
    
    def __getitem__(self, name: str) -> LoadedModel:
        return self._models[name]
