# RWKVServe

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**高性能 RWKV 推理服务框架，对标 vLLM 设计，提供 OpenAI 兼容 API 与连续批处理能力。**

<p align="center">
  <a href="README.md">English</a>
</p>

## 特性

- **Continuous Batching** — 基于 SchedulerCore 的动态调度，短请求不会被长请求阻塞，支持 Chunked Prefill 控制显存峰值
- **OpenAI 兼容 API** — 完整实现 `/v1/chat/completions` 与 `/v1/completions`，可直接对接 OpenAI SDK
- **State Cache** — 基于 Trie 的前缀级状态缓存，加速重复前缀推理
- **LoRA 热加载** — 支持加载 LoRA 适配器并在线推理（vLLM 风格的 `--enable-lora --lora-modules name=path`）
- **Reasoning 输出** — 支持思考模式（`<think>...</think>`），分离推理过程与最终回答，兼容 `reasoning_content` 字段
- **Data Parallel** — 多 GPU 数据并行推理，自动负载均衡
- **Multi-Model** — 同时部署多个模型，按 `model` 字段自动路由
- **Structured Output** — 通过 JSON Schema 约束生成输出格式
- **vLLM 风格 Python API** — `LLM.generate()` 离线批量推理，支持任意数量 prompt 的连续批处理
- **API Key 鉴权** — 支持多 Key 认证，兼容环境变量配置

## 安装

```bash
# 从源码安装
pip install -e .

# 安装结构化输出支持
pip install -e ".[structured-output]"

# 安装全部依赖（含开发工具）
pip install -e ".[all]"
```

## 快速开始

### 1. 启动 API 服务

```bash
# 单模型
rwkvserve --model-path /path/to/model --max-batch-size 32

# 指定模型名称和精度
rwkvserve --model-path /path/to/model --model-name rwkv-7 --dtype bf16

# 多模型部署
rwkvserve \
    --model model1:/path/to/model1 \
    --model model2:/path/to/model2:cuda:0

# 数据并行（多 GPU）
rwkvserve --model model1:/path/to/model1 --gpus 0,1,2,3
```

### 2. 加载 LoRA 适配器

```bash
rwkvserve \
    --model-path /path/to/base_model \
    --enable-lora \
    --lora-modules my-lora=/path/to/lora_adapter
```

LoRA 权重在启动时合并到基础模型中，无额外推理开销。API 请求通过 `model` 字段选择适配器名称（如 `"my-lora"`）。

### 3. 启用 Reasoning 模式

```bash
rwkvserve \
    --model-path /path/to/model \
    --enable-reasoning --reasoning-parser deepseek_r1
```

启用后，模型输出中的 `<think>...</think>` 内容会自动提取到 `reasoning_content` 字段，与 vLLM 的 reasoning 输出保持一致。

### 4. 使用 OpenAI SDK 调用

```python
from openai import OpenAI

client = OpenAI(api_key="dummy", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="rwkv-7",
    messages=[{"role": "user", "content": "你好"}],
)
print(response.choices[0].message.content)

# 访问 reasoning_content（需服务端启用 --enable-reasoning）
msg = response.choices[0].message
if hasattr(msg, "reasoning_content") and msg.reasoning_content:
    print("思考过程:", msg.reasoning_content)
print("回答:", msg.content)
```

### 5. Python 离线批量推理（LLM.generate）

```python
from rwkvserve import LLM, SamplingParams

# 基础模型推理
llm = LLM(model="/path/to/model", max_batch_size=256, dtype="bf16")

# 带 LoRA 适配器
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

### 6. 命令行交互推理

```bash
# 单次推理
rwkvserve-infer --model /path/to/model --prompt "你好" --stream

# 交互式对话
rwkvserve-infer --model /path/to/model --chat
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/v1/models` | 列出可用模型 |
| POST | `/v1/chat/completions` | 对话补全（支持流式） |
| POST | `/v1/completions` | 文本补全（支持流式） |
| GET | `/health` | 健康检查 |
| GET | `/docs` | Swagger API 文档 |

### 请求示例

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

## CLI 参数速查

```
rwkvserve [选项]

模型参数:
  --model-path PATH           模型目录路径
  --model-name NAME           API 中的模型名称（默认: rwkv-7）
  --model NAME:PATH[:DEVICE]  多模型配置（可多次使用）
  --model-config FILE         YAML 格式模型配置文件

LoRA 参数:
  --enable-lora               启用 LoRA 适配器
  --lora-modules NAME=PATH    LoRA 模块（可多次使用）

Reasoning 参数:
  --enable-reasoning          启用 reasoning 内容提取
  --reasoning-parser NAME     解析器名称（默认: deepseek_r1）

运行参数:
  --device {auto,cuda,cpu}    计算设备（默认: auto）
  --dtype {fp32,fp16,bf16}    模型精度
  --max-batch-size N          最大批大小（默认: 32）
  --prefill-chunk-size N      Chunked Prefill 块大小（默认: 512）

服务参数:
  --host HOST                 监听地址（默认: 0.0.0.0）
  --port PORT                 监听端口（默认: 8000）
  --gpus IDS                  数据并行 GPU 列表（如 0,1,2,3）
  --stop                      停止运行中的服务并清理资源
  --api-key KEY               API Key 鉴权（可多次使用）

State Cache 参数:
  --max-cache-memory GB       状态缓存内存上限（默认: 4.0）
  --cache-level LEVEL         缓存级别: none / exact / prefix（默认: prefix）
```

## 项目结构

```
rwkvserve/
├── models/            # RWKV 模型实现（RWKV-7）
│   └── rwkv7/         #   模型定义、配置、CUDA 算子
├── inference/         # 推理引擎
│   ├── scheduler_core.py    # Continuous Batching 调度器
│   ├── state_cache.py       # Trie-based State Cache
│   ├── pipeline.py          # 推理管线
│   └── structured_output.py # 结构化输出约束
├── api/               # OpenAI 兼容 API 服务
│   ├── api_server.py        # FastAPI 应用
│   ├── async_serving_chat.py      # Chat Completions
│   ├── async_serving_completion.py # Text Completions
│   ├── model_manager.py    # 多模型管理与路由
│   └── protocol.py         # 请求/响应协议定义
├── entrypoints/       # 入口
│   └── llm.py         #   LLM.generate() 离线批量推理
├── reasoning/         # Reasoning 输出解析
│   ├── base.py        #   解析器抽象基类与注册管理
│   └── deepseek_r1.py #   <think>...</think> 解析器
├── peft.py            # LoRA 适配器加载与权重合并
├── sampling_params.py # 采样参数（vLLM 风格）
├── outputs.py         # 输出类型定义
├── cli/               # CLI 工具
│   ├── serve.py       #   rwkvserve 命令
│   └── infer.py       #   rwkvserve-infer 命令
└── data/tokenizers/   # Tokenizer 实现
```

## 示例脚本

`examples/` 目录下提供了完整的使用示例：

| 脚本 | 说明 |
|------|------|
| `start_server.sh` | 启动 API 服务（含 LoRA 和 Reasoning 配置） |
| `test_server.sh` | 用 curl 测试各 API 端点 |
| `test_openai_sdk.py` | 使用 OpenAI SDK 测试对话推理 |
| `test_llm_generate.py` | 使用 LLM.generate() 测试离线批量推理 |

## 许可证

本项目基于 Apache License 2.0 许可。详见 [LICENSE](LICENSE)。

## 致谢

- 基于 [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) 官方实现
- API 设计对标 [vLLM](https://github.com/vllm-project/vllm)
- 基于 [FastAPI](https://fastapi.tiangolo.com/) 和 [PyTorch](https://pytorch.org/) 构建
