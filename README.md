# RWKVServe

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**High-performance RWKV inference and serving framework, aligned with vLLM design, providing an OpenAI-compatible API with Continuous Batching.**

<p align="center">
  <a href="README_zh.md">中文文档</a>
</p>

## Features

- **Continuous Batching** — Dynamic scheduling via SchedulerCore; short requests are never blocked by long ones; Chunked Prefill to control peak memory
- **OpenAI-Compatible API** — Full implementation of `/v1/chat/completions` and `/v1/completions`, works directly with the OpenAI SDK
- **State Cache** — Trie-based prefix-level state caching for accelerated repeated-prefix inference
- **LoRA Adapter** — Load LoRA adapters and serve them online (vLLM-style `--enable-lora --lora-modules name=path`)
- **Reasoning Output** — Thinking mode support (`<think>...</think>`), separates reasoning from the final answer via `reasoning_content` field
- **Data Parallel** — Multi-GPU data-parallel inference with automatic load balancing
- **Multi-Model** — Serve multiple models simultaneously, auto-routed by the `model` field
- **Structured Output** — JSON Schema enforcement for constrained generation
- **vLLM-style Python API** — `LLM.generate()` for offline batch inference with Continuous Batching over arbitrary number of prompts
- **API Key Auth** — Multi-key authentication, configurable via CLI or environment variable

## Installation

```bash
# Install from source
pip install -e .

# With structured output support
pip install -e ".[structured-output]"

# With all extras (dev tools included)
pip install -e ".[all]"
```

## Quick Start

### 1. Start the API Server

```bash
# Single model
rwkvserve --model-path /path/to/model --max-batch-size 32

# With model name and dtype
rwkvserve --model-path /path/to/model --model-name rwkv-7 --dtype bf16

# Multi-model deployment
rwkvserve \
    --model model1:/path/to/model1 \
    --model model2:/path/to/model2:cuda:0

# Data parallel (multi-GPU)
rwkvserve --model model1:/path/to/model1 --gpus 0,1,2,3
```

### 2. Serve with LoRA Adapter

```bash
rwkvserve \
    --model-path /path/to/base_model \
    --enable-lora \
    --lora-modules my-lora=/path/to/lora_adapter
```

LoRA weights are merged into the base model at startup — zero runtime overhead. API requests select the adapter by its name via the `model` field (e.g., `"my-lora"`).

### 3. Enable Reasoning Mode

```bash
rwkvserve \
    --model-path /path/to/model \
    --enable-reasoning --reasoning-parser deepseek_r1
```

When enabled, `<think>...</think>` content in model output is automatically extracted into the `reasoning_content` field, consistent with vLLM's reasoning output.

### 4. Call with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(api_key="dummy", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="rwkv-7",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

# Access reasoning_content (requires --enable-reasoning on server)
msg = response.choices[0].message
if hasattr(msg, "reasoning_content") and msg.reasoning_content:
    print("Thinking:", msg.reasoning_content)
print("Answer:", msg.content)
```

### 5. Offline Batch Inference (LLM.generate)

```python
from rwkvserve import LLM, SamplingParams

# Basic
llm = LLM(model="/path/to/model", max_batch_size=256, dtype="bf16")

# With LoRA adapter
llm = LLM(
    model="/path/to/base_model",
    enable_lora=True,
    lora_path="/path/to/lora_adapter",
    dtype="bf16",
)

params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
outputs = llm.generate(["Hello, world!"] * 1000, params, use_tqdm=True)

for output in outputs:
    print(output.outputs[0].text)
```

### 6. Command-line Inference

```bash
# Single prompt
rwkvserve-infer --model /path/to/model --prompt "Hello!" --stream

# Interactive chat
rwkvserve-infer --model /path/to/model --chat
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | Chat completion (streaming supported) |
| POST | `/v1/completions` | Text completion (streaming supported) |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger API docs |

### Request Example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv-7",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256,
    "temperature": 0.8,
    "stream": false
  }'
```

## CLI Reference

```
rwkvserve [options]

Model:
  --model-path PATH           Path to model directory
  --model-name NAME           Model name in API (default: rwkv-7)
  --model NAME:PATH[:DEVICE]  Multi-model config (repeatable)
  --model-config FILE         YAML model config file

LoRA:
  --enable-lora               Enable LoRA adapter support
  --lora-modules NAME=PATH    LoRA module to load (repeatable)

Reasoning:
  --enable-reasoning          Enable reasoning content extraction
  --reasoning-parser NAME     Parser name (default: deepseek_r1)

Runtime:
  --device {auto,cuda,cpu}    Compute device (default: auto)
  --dtype {fp32,fp16,bf16}    Model precision
  --max-batch-size N          Max batch size (default: 32)
  --prefill-chunk-size N      Chunked prefill block size (default: 512)

Server:
  --host HOST                 Listen address (default: 0.0.0.0)
  --port PORT                 Listen port (default: 8000)
  --gpus IDS                  Data-parallel GPU list (e.g. 0,1,2,3)
  --stop                      Stop running service and clean up resources
  --api-key KEY               API key for auth (repeatable)

State Cache:
  --max-cache-memory GB       State cache memory limit (default: 4.0)
  --cache-level LEVEL         Cache level: none / exact / prefix (default: prefix)
```

## Project Structure

```
rwkvserve/
├── models/            # RWKV model implementation (RWKV-7)
│   └── rwkv7/         #   Model definition, config, CUDA operators
├── inference/         # Inference engine
│   ├── scheduler_core.py    # Continuous Batching scheduler
│   ├── state_cache.py       # Trie-based State Cache
│   ├── pipeline.py          # Inference pipeline
│   └── structured_output.py # Structured output enforcement
├── api/               # OpenAI-compatible API server
│   ├── api_server.py        # FastAPI application
│   ├── async_serving_chat.py      # Chat completions handler
│   ├── async_serving_completion.py # Text completions handler
│   ├── model_manager.py    # Multi-model management & routing
│   └── protocol.py         # Request / response protocol
├── entrypoints/       # Entrypoints
│   └── llm.py         #   LLM.generate() offline batch inference
├── reasoning/         # Reasoning output parsing
│   ├── base.py        #   Abstract parser & registry
│   └── deepseek_r1.py #   <think>...</think> parser
├── peft.py            # LoRA adapter loading & weight merging
├── sampling_params.py # Sampling parameters (vLLM-style)
├── outputs.py         # Output type definitions
├── cli/               # CLI tools
│   ├── serve.py       #   rwkvserve command
│   └── infer.py       #   rwkvserve-infer command
└── data/tokenizers/   # Tokenizer implementations
```

## Examples

The `examples/` directory provides ready-to-use scripts:

| Script | Description |
|--------|-------------|
| `start_server.sh` | Start the API server with LoRA and Reasoning config |
| `test_server.sh` | Test API endpoints with curl |
| `test_openai_sdk.py` | Test chat inference with OpenAI SDK |
| `test_llm_generate.py` | Test offline batch inference with LLM.generate() |

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the official [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) implementation
- API design aligned with [vLLM](https://github.com/vllm-project/vllm)
- Built with [FastAPI](https://fastapi.tiangolo.com/) and [PyTorch](https://pytorch.org/)
