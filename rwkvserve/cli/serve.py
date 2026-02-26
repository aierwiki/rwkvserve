"""
API server CLI tool - supports multi-model deployment and data parallelism
"""

import argparse
import sys
import os
import signal
import time
import gc
from typing import List, Optional
import torch

from rwkvserve.api.api_server import create_app
from rwkvserve.api.model_manager import ModelConfig, ModelManager


def stop_service(port: int = 8000, force: bool = False, verbose: bool = True) -> int:
    """
    Stop the rwkvserve service and clean up resources.

    Args:
        port: Service port (will clean port through port+10)
        force: Skip graceful shutdown and force kill
        verbose: Print detailed information

    Returns:
        0 on success, 1 on failure
    """
    import psutil

    found_any = False
    killed_pids = set()
    current_pid = os.getpid()

    def vprint(msg):
        if verbose:
            print(msg, flush=True)

    vprint("Checking for running services...")

    def is_target_process(cmdline: str, pid: int) -> bool:
        if pid == current_pid:
            return False
        if "--stop" in cmdline:
            return False
        if "rwkvserve.cli.serve" in cmdline:
            return True
        if "uvicorn" in cmdline and "rwkvserve" in cmdline:
            return True
        return False

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = " ".join(proc.info['cmdline'] or [])
            pid = proc.info['pid']

            if is_target_process(cmdline, pid):
                found_any = True

                if pid in killed_pids:
                    continue

                if not force:
                    vprint(f"  Sending SIGTERM to PID: {pid}")
                    try:
                        os.kill(pid, signal.SIGTERM)
                        killed_pids.add(pid)
                    except (ProcessLookupError, PermissionError):
                        pass
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if found_any and not force:
        vprint("  Waiting for graceful shutdown...")
        time.sleep(3)

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = " ".join(proc.info['cmdline'] or [])
            pid = proc.info['pid']

            if is_target_process(cmdline, pid) and pid not in killed_pids:
                vprint(f"  Force killing PID: {pid}")
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed_pids.add(pid)
                except (ProcessLookupError, PermissionError):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    for p in range(port, port + 11):
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr.port == p and conn.pid:
                pid = conn.pid
                if pid not in killed_pids:
                    found_any = True
                    vprint(f"  Port {p} occupied by PID {pid}, stopping...")
                    try:
                        os.kill(pid, signal.SIGKILL)
                        killed_pids.add(pid)
                    except (ProcessLookupError, PermissionError):
                        pass

    if killed_pids:
        vprint("  Waiting for processes to exit...")
        time.sleep(2)

    vprint("Cleaning GPU cache...")
    try:
        gc.collect()
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            vprint(f"  Cleared cache on {num_gpus} GPU(s)")
    except Exception as e:
        vprint(f"  Warning: failed to clear GPU cache: {e}")

    remaining = []
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = " ".join(proc.info['cmdline'] or [])
            pid = proc.info['pid']
            if is_target_process(cmdline, pid):
                remaining.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if remaining:
        vprint(f"Warning: remaining processes: {remaining}")
        vprint("  Please check manually or use --force")
        return 1
    else:
        if found_any or killed_pids:
            vprint(f"Stopped {len(killed_pids)} process(es), service fully cleaned up")
        else:
            vprint("No running services found")
        return 0


def main():
    """Entry point for rwkvserve command (multi-model support)."""
    parser = argparse.ArgumentParser(
        description="Launch RWKVServe OpenAI-compatible API server (multi-model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
========== Usage ==========

Option 1: Single model (backward compatible)
  rwkvserve --model-path /path/to/model

Option 2: Multiple models (CLI)
  rwkvserve \\
      --model model1:/path/to/model1 \\
      --model model2:/path/to/model2:cuda:0 \\
      --model model3:/path/to/model3:cuda:1

Option 3: Config file
  rwkvserve --model-config models.yaml

Option 4: Data parallel (multi-GPU)
  rwkvserve --model model1:/path/to/model1 --gpus 0,1,2,3

========== Model format ==========

--model argument format: name:path[:device]

Examples:
  --model gpt:/models/gpt                  # Use default device
  --model chat:/models/chat:cuda           # Use CUDA
  --model fast:/models/fast:cuda:0         # Use cuda:0
  --model cpu-model:/models/cpu:cpu        # Use CPU

========== Data parallel mode ==========

Use --gpus to enable data-parallel inference:
  rwkvserve --model model1:/path/to/model1 --gpus 0,1,2,3

Features:
  - Single API service
  - Automatic request distribution across GPUs
  - Load balancing: least-connections strategy
  - Independent State Cache per GPU

========== API endpoints ==========

  GET  /v1/models              - List all available models
  POST /v1/chat/completions    - Chat completions (routed by model field)
  POST /v1/completions         - Text completions (routed by model field)
  GET  /health                 - Health check
  GET  /docs                   - API docs (Swagger UI)

========== API Key authentication ==========

  # Enable API Key auth
  rwkvserve --model-path /path/to/model --api-key your-secret-key

  # Multiple API Keys
  rwkvserve --model-path /path/to/model --api-key key1 --api-key key2

  # Using environment variable
  export RWKV_API_KEY=your-secret-key
  rwkvserve --model-path /path/to/model

========== Test the API ==========

  # Using curl (with API Key)
  curl http://localhost:8000/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -H "Authorization: Bearer your-secret-key" \\
    -d '{"model": "model1", "messages": [{"role": "user", "content": "Hello!"}]}'

  # Using OpenAI Python SDK (with API Key)
  from openai import OpenAI
  client = OpenAI(
      api_key="your-secret-key",
      base_url="http://localhost:8000/v1"
  )

  response = client.chat.completions.create(
      model="model1",
      messages=[{"role": "user", "content": "Hello!"}],
  )
  print(response.choices[0].message.content)
""",
    )

    model_group = parser.add_argument_group("Model arguments")

    model_group.add_argument(
        "--model",
        type=str,
        action="append",
        dest="models",
        metavar="NAME:PATH[:DEVICE]",
        help="Model config, format: name:path[:device]. Can be used multiple times",
    )

    model_group.add_argument(
        "--model-config",
        type=str,
        help="Model config file path (YAML format)",
    )

    model_group.add_argument(
        "--model-path",
        type=str,
        help="Model directory path (backward compatible, use --model instead)",
    )
    model_group.add_argument(
        "--model-name",
        type=str,
        default="rwkv-7",
        help="Model name in API (only for --model-path, default: rwkv-7)",
    )

    lora_group = parser.add_argument_group("LoRA arguments")
    lora_group.add_argument(
        "--enable-lora",
        action="store_true",
        default=False,
        help="Enable LoRA adapter support",
    )
    lora_group.add_argument(
        "--lora-modules",
        type=str,
        action="append",
        metavar="NAME=PATH",
        help=(
            "LoRA module to load, format: name=path. Can be used multiple times. "
            "Requests select the adapter via the 'model' field. "
            "Example: --lora-modules my-lora=/path/to/adapter"
        ),
    )

    reasoning_group = parser.add_argument_group("Reasoning arguments")
    reasoning_group.add_argument(
        "--enable-reasoning",
        action="store_true",
        default=False,
        help="Enable reasoning content extraction (separates <think>...</think> from response)",
    )
    reasoning_group.add_argument(
        "--reasoning-parser",
        type=str,
        default="deepseek_r1",
        help="Reasoning parser to use (default: deepseek_r1). Available: deepseek_r1, think",
    )

    runtime_group = parser.add_argument_group("Runtime arguments")
    runtime_group.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Default compute device (default: auto)",
    )
    runtime_group.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        help="Model data type (default: original weight precision)",
    )
    runtime_group.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Maximum batch size (default: 32). Set > 1 to enable Continuous Batching",
    )
    runtime_group.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=512,
        help="Chunked prefill block size (default: 512). Controls peak memory for long prompts",
    )

    server_group = parser.add_argument_group("Server arguments")
    server_group.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host address (default: 0.0.0.0)",
    )
    server_group.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    server_group.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    server_group.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Data parallel mode: GPU ID list (e.g. 0,1,2,3). "
             "Automatically distributes requests across GPUs",
    )
    server_group.add_argument(
        "--stop",
        action="store_true",
        help="Stop running service and clean up resources",
    )
    server_group.add_argument(
        "--force",
        action="store_true",
        help="Use with --stop to skip graceful shutdown and force kill",
    )
    server_group.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development mode)",
    )

    security_group = parser.add_argument_group("Security arguments")
    security_group.add_argument(
        "--api-key",
        type=str,
        action="append",
        dest="api_keys",
        metavar="KEY",
        help="API Key (can be used multiple times). Also settable via RWKV_API_KEY env var",
    )

    log_group = parser.add_argument_group("Logging arguments")
    log_group.add_argument(
        "--log-requests",
        type=str,
        metavar="DIR",
        help="Request log directory. Saves all request inputs, params and outputs",
    )

    cache_group = parser.add_argument_group("State Cache arguments")
    cache_group.add_argument(
        "--max-cache-memory",
        type=float,
        default=4.0,
        help="State cache max memory in GB (default: 4.0)",
    )
    cache_group.add_argument(
        "--cache-level",
        type=str,
        default="prefix",
        choices=["none", "exact", "prefix"],
        help="Cache level: none, exact, prefix (default: prefix)",
    )
    cache_group.add_argument(
        "--no-cache-first-message",
        action="store_true",
        help="Disable caching of the first message (system prompt)",
    )

    args = parser.parse_args()

    if args.stop:
        print("Stopping rwkvserve service...")
        return stop_service(port=args.port, force=args.force, verbose=True)

    if args.device == "auto":
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        default_device = args.device

    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
        if len(gpu_ids) < 1:
            print("Error: --gpus requires at least one GPU")
            return 1
        print(f"\nData parallel mode: {len(gpu_ids)} GPU(s)")
        print(f"  GPU IDs: {gpu_ids}")

    model_configs = []

    if args.model_config:
        if not os.path.exists(args.model_config):
            print(f"Error: config file not found: {args.model_config}")
            return 1

        print(f"Loading models from config: {args.model_config}")

        try:
            import yaml

            with open(args.model_config, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            models_data = config_data.get("models", [])
            if not models_data:
                print("Error: no model configurations found in config file")
                return 1

            model_configs = [ModelConfig.from_dict(m) for m in models_data]

        except Exception as e:
            print(f"Error: failed to parse config file: {e}")
            import traceback
            traceback.print_exc()
            return 1

    elif args.models:
        for model_str in args.models:
            try:
                config = ModelConfig.from_string(model_str)
                if config.device == "auto":
                    config.device = default_device
                if config.dtype is None and args.dtype:
                    config.dtype = args.dtype
                model_configs.append(config)
            except ValueError as e:
                print(f"Error: {e}")
                return 1

    elif args.model_path:
        if not os.path.exists(args.model_path):
            print(f"Error: model path does not exist: {args.model_path}")
            return 1

        model_configs = [ModelConfig(
            name=args.model_name,
            path=args.model_path,
            device=default_device,
            dtype=args.dtype,
        )]

    else:
        print("Error: model configuration required")
        print("Usage:")
        print("  --model name:path[:device]    (can be used multiple times)")
        print("  --model-config config.yaml")
        print("  --model-path path             (backward compatible)")
        print()
        print("Run 'rwkvserve --help' for details")
        return 1

    # Parse LoRA modules (vLLM-compatible: --enable-lora --lora-modules name=path)
    lora_specs = None
    if args.enable_lora:
        if not args.lora_modules:
            print("Error: --enable-lora requires at least one --lora-modules")
            return 1
        from rwkvserve.api.model_manager import LoRAModuleSpec
        lora_specs = []
        for spec_str in args.lora_modules:
            if "=" not in spec_str:
                print(f"Error: invalid --lora-modules format: {spec_str!r}")
                print("Expected: name=path  (e.g. my-lora=/path/to/adapter)")
                return 1
            name, path = spec_str.split("=", 1)
            if not os.path.exists(path):
                print(f"Error: LoRA adapter path does not exist: {path}")
                return 1
            lora_specs.append(LoRAModuleSpec(name=name.strip(), path=path.strip()))
        print(f"\nLoRA adapters: {len(lora_specs)}")
        for spec in lora_specs:
            print(f"  {spec.name} -> {spec.path}")

    for config in model_configs:
        if args.max_batch_size > 1:
            config.max_batch_size = args.max_batch_size
        config.prefill_chunk_size = args.prefill_chunk_size

        if gpu_ids:
            config.gpu_ids = gpu_ids

        if lora_specs:
            config.enable_lora = True
            config.lora_modules = lora_specs

    api_keys = args.api_keys or []
    log_requests = args.log_requests

    cache_config = {
        "max_cache_memory_gb": args.max_cache_memory,
        "cache_level": args.cache_level,
        "cache_first_message": not args.no_cache_first_message,
    }

    print("=" * 70)
    print("  RWKVServe OpenAI-Compatible API Server")
    print("=" * 70)
    print(f"Models to load: {len(model_configs)}")
    for i, config in enumerate(model_configs, 1):
        print(f"  {i}. {config.name}")
        print(f"     Path: {config.path}")
        if config.is_data_parallel:
            print(f"     Mode: data parallel (GPUs: {config.gpu_ids})")
        else:
            print(f"     Device: {config.device}")
    if args.enable_lora and lora_specs:
        print(f"LoRA adapters:")
        for spec in lora_specs:
            print(f"  - {spec.name} ({spec.path})")
    print(f"Server: http://{args.host}:{args.port}")
    if api_keys:
        print(f"API Key auth: enabled ({len(api_keys)} key(s))")
    else:
        if os.environ.get("RWKV_API_KEY"):
            print("API Key auth: enabled (env var)")
        else:
            print("API Key auth: disabled")
    if log_requests:
        print(f"Request logging: {log_requests}")
    # Reasoning parser
    reasoning_parser = None
    if args.enable_reasoning:
        from rwkvserve.reasoning import ReasoningParserManager
        reasoning_parser = ReasoningParserManager.get_parser(args.reasoning_parser)
        print(f"Reasoning: enabled (parser: {args.reasoning_parser})")
    else:
        print("Reasoning: disabled")

    print("State Cache:")
    print(f"  Memory limit: {cache_config['max_cache_memory_gb']:.1f} GB")
    print(f"  Cache level: {cache_config['cache_level']}")
    print(f"  Cache first message: {'yes' if cache_config['cache_first_message'] else 'no'}")
    print("=" * 70)

    try:
        import uvicorn

        app = create_app(
            models=model_configs,
            api_keys=api_keys if api_keys else None,
            log_requests=log_requests,
            cache_config=cache_config,
            reasoning_parser=reasoning_parser,
        )

        print("\nStarting server...\n")
        print(f"API base URL: http://{args.host}:{args.port}/v1")
        print(f"API docs: http://{args.host}:{args.port}/docs")
        print(f"Health check: http://{args.host}:{args.port}/health")
        if gpu_ids:
            print(f"DP stats: http://{args.host}:{args.port}/dp/stats")
        print("\nPress Ctrl+C to stop\n")

        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers if not args.reload else 1,
            reload=args.reload,
            log_level="info",
        )

    except ImportError:
        print("\nError: missing dependency 'uvicorn'")
        print("Install it with: pip install uvicorn")
        return 1

    except KeyboardInterrupt:
        print("\n\nServer stopped")
        return 0

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
