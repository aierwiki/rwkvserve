#!/usr/bin/env python3
"""
RWKVServe - OpenAI SDK Test Script

Sends requests to a running RWKVServe API server using the OpenAI Python SDK,
saves all input/output pairs to a JSON file for review.

Test data: sharegpt_roleplay_sample_100.jsonl (same as rwkvtune test_generation.py)

Usage:
    python test_openai_sdk.py
    python test_openai_sdk.py --num-samples 5 --server-url http://localhost:8000
"""

import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI


def load_test_samples(data_file: str, num_samples: int) -> list:
    """Load test samples from ShareGPT-format JSONL file."""
    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def sharegpt_to_messages(conversations: list) -> tuple:
    """
    Convert ShareGPT conversation to OpenAI messages format.

    Returns (messages, reference) where:
    - messages: list of dicts with role/content for the API request
      (system + all user/assistant turns up to the last user turn)
    - reference: the first assistant reply from training data (for comparison)
    """
    messages = []
    reference = ""

    role_map = {
        "system": "system",
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
    }

    for turn in conversations:
        role = role_map.get(turn.get("from", "").lower())
        content = turn.get("value", "")
        if role is None:
            continue
        messages.append({"role": role, "content": content})

    # Find the first assistant reply as reference
    for msg in messages:
        if msg["role"] == "assistant":
            reference = msg["content"]
            break

    # Trim: keep messages up to (and including) the first user message only,
    # so the model generates a fresh reply.
    trimmed = []
    for msg in messages:
        if msg["role"] in ("system", "user"):
            trimmed.append(msg)
            if msg["role"] == "user":
                break

    return trimmed, reference


def main():
    parser = argparse.ArgumentParser(description="RWKVServe OpenAI SDK Test")
    parser.add_argument(
        "--server-url", type=str, default="http://localhost:8000",
        help="Server base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model", type=str, default="rwkv-7-grpo",
        help="Model name to use in requests (default: rwkv-7-grpo)",
    )
    parser.add_argument(
        "--data-file", type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "../../rwkvtune/examples/data/sharegpt_roleplay_sample_100.jsonl",
        ),
        help="Test data file (ShareGPT JSONL format)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=5,
        help="Number of test samples (default: 5)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="Max tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9,
        help="Top-p sampling (default: 0.9)",
    )
    parser.add_argument(
        "--output", type=str, default="test_openai_sdk_results.json",
        help="Output JSON file (default: test_openai_sdk_results.json)",
    )
    args = parser.parse_args()

    data_file = os.path.abspath(args.data_file)
    if not os.path.exists(data_file):
        print(f"Error: test data file not found: {data_file}")
        return 1

    client = OpenAI(api_key="none", base_url=f"{args.server_url}/v1")

    print("=" * 60)
    print("  RWKVServe - OpenAI SDK Test")
    print("=" * 60)
    print(f"  Server:       {args.server_url}")
    print(f"  Model:        {args.model}")
    print(f"  Data file:    {data_file}")
    print(f"  Samples:      {args.num_samples}")
    print(f"  Max tokens:   {args.max_tokens}")
    print(f"  Temperature:  {args.temperature}")
    print(f"  Top-p:        {args.top_p}")
    print(f"  Output:       {args.output}")
    print("=" * 60)

    # Verify server is reachable
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        print(f"\nAvailable models: {available}")
        if args.model not in available:
            print(f"Warning: model '{args.model}' not in available models")
    except Exception as e:
        print(f"\nError: cannot connect to server: {e}")
        return 1

    # Load test data
    samples = load_test_samples(data_file, args.num_samples)
    print(f"Loaded {len(samples)} test samples\n")

    results = []

    for idx, sample in enumerate(samples):
        conversations = sample.get("conversations", [])
        if not conversations:
            continue

        messages, reference = sharegpt_to_messages(conversations)
        if not messages:
            continue

        print(f"{'─' * 60}")
        print(f"[Sample {idx + 1}/{len(samples)}]")
        print(f"{'─' * 60}")

        # Show input
        for msg in messages:
            role_label = msg["role"].upper()
            content_preview = msg["content"][:200]
            if len(msg["content"]) > 200:
                content_preview += "..."
            print(f"  {role_label}: {content_preview}")

        # Call API
        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            elapsed = time.time() - t0

            msg = response.choices[0].message
            content = msg.content
            reasoning_content = getattr(msg, "reasoning_content", None)
            usage = response.usage
            finish_reason = response.choices[0].finish_reason

            if reasoning_content:
                print(f"\n  THINKING: {reasoning_content}")
                print(f"\n  ANSWER: {content}")
            else:
                print(f"\n  ASSISTANT: {content}")

            print(f"\n  [tokens: prompt={usage.prompt_tokens}, "
                  f"completion={usage.completion_tokens}, "
                  f"finish={finish_reason}, "
                  f"time={elapsed:.2f}s]")

            if reference:
                ref_preview = reference[:200] + ("..." if len(reference) > 200 else "")
                print(f"  REFERENCE: {ref_preview}")

            record = {
                "index": idx,
                "input_messages": messages,
                "output": content,
                "reference": reference,
                "finish_reason": finish_reason,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                },
                "elapsed_seconds": round(elapsed, 3),
            }
            if reasoning_content:
                record["reasoning_content"] = reasoning_content
            results.append(record)

        except Exception as e:
            print(f"\n  ERROR: {e}")
            results.append({
                "index": idx,
                "input_messages": messages,
                "output": None,
                "reference": reference,
                "error": str(e),
            })

        print()

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"  Done! {len(results)} results saved to: {output_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
