#!/usr/bin/env python3
"""
RWKVServe - LLM.generate() Test Script

Tests offline batch inference using the vLLM-style LLM.generate() API
with a base model + LoRA adapter. Saves input/output pairs for review.

Uses the same test data as rwkvtune's test_generation.py.

Usage:
    CUDA_VISIBLE_DEVICES=6 python test_llm_generate.py
    CUDA_VISIBLE_DEVICES=6 python test_llm_generate.py --num-samples 10
"""

import argparse
import json
import os
import time


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


def sharegpt_to_prompt(conversations: list) -> tuple:
    """
    Convert ShareGPT conversation to a prompt string and extract reference.

    Returns (prompt, reference):
    - prompt: formatted string for model input (system + first user turn)
    - reference: the first assistant reply from training data (for comparison)
    """
    system_parts = []
    first_user = None
    reference = ""

    for turn in conversations:
        role = turn.get("from", "").lower()
        content = turn.get("value", "")

        if role == "system":
            system_parts.append(content.strip())
        elif role in ("human", "user"):
            if first_user is None:
                first_user = content.strip()
        elif role in ("gpt", "assistant"):
            if not reference:
                reference = content

    if first_user is None:
        return "", reference

    prompt_parts = []
    if system_parts:
        prompt_parts.append(f"System: {' '.join(system_parts)}\n\n")
    prompt_parts.append(f"User: {first_user}\n\nAssistant:")

    return "".join(prompt_parts), reference


def main():
    parser = argparse.ArgumentParser(description="RWKVServe LLM.generate() Test")
    parser.add_argument(
        "--base-model", type=str,
        default="output_sft_grpo/rwkv7-epoch3",
        help="Base model path",
    )
    parser.add_argument(
        "--lora-path", type=str,
        default="output_grpo/rwkv7-batch1728",
        help="LoRA adapter path (set to empty string to disable)",
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
        "--dtype", type=str, default="bf16",
        help="Model precision (default: bf16)",
    )
    parser.add_argument(
        "--output", type=str, default="test_llm_generate_results.json",
        help="Output JSON file (default: test_llm_generate_results.json)",
    )
    args = parser.parse_args()

    data_file = os.path.abspath(args.data_file)
    if not os.path.exists(data_file):
        print(f"Error: test data file not found: {data_file}")
        return 1

    enable_lora = bool(args.lora_path)

    print("=" * 60)
    print("  RWKVServe - LLM.generate() Test")
    print("=" * 60)
    print(f"  Base model:   {args.base_model}")
    if enable_lora:
        print(f"  LoRA adapter: {args.lora_path}")
    print(f"  Data file:    {data_file}")
    print(f"  Samples:      {args.num_samples}")
    print(f"  Max tokens:   {args.max_tokens}")
    print(f"  Temperature:  {args.temperature}")
    print(f"  Top-p:        {args.top_p}")
    print(f"  Dtype:        {args.dtype}")
    print(f"  Output:       {args.output}")
    print("=" * 60)

    # Load test data first (before loading model to fail fast)
    samples = load_test_samples(data_file, args.num_samples)
    print(f"\nLoaded {len(samples)} test samples")

    # Build prompts
    prompts = []
    references = []
    valid_indices = []
    for idx, sample in enumerate(samples):
        conversations = sample.get("conversations", [])
        if not conversations:
            continue
        prompt, reference = sharegpt_to_prompt(conversations)
        if not prompt:
            continue
        prompts.append(prompt)
        references.append(reference)
        valid_indices.append(idx)

    print(f"Built {len(prompts)} prompts\n")

    # Initialize LLM
    from rwkvserve import LLM, SamplingParams

    llm = LLM(
        model=args.base_model,
        dtype=args.dtype,
        enable_lora=enable_lora,
        lora_path=args.lora_path if enable_lora else None,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Run batch inference
    print(f"\nRunning LLM.generate() on {len(prompts)} prompts...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    total_elapsed = time.time() - t0

    # Process results
    results = []
    for i, output in enumerate(outputs):
        idx = valid_indices[i]
        prompt = prompts[i]
        reference = references[i]
        generated = output.outputs[0].text
        finish_reason = output.outputs[0].finish_reason
        num_tokens = len(output.outputs[0].token_ids)

        print(f"\n{'─' * 60}")
        print(f"[Sample {idx + 1}]")
        print(f"{'─' * 60}")

        prompt_preview = prompt[:300] + ("..." if len(prompt) > 300 else "")
        print(f"  PROMPT: {prompt_preview}")
        print(f"\n  OUTPUT: {generated}")
        print(f"\n  [tokens: {num_tokens}, finish: {finish_reason}]")

        if reference:
            ref_preview = reference[:200] + ("..." if len(reference) > 200 else "")
            print(f"  REFERENCE: {ref_preview}")

        record = {
            "index": idx,
            "prompt": prompt,
            "output": generated,
            "reference": reference,
            "finish_reason": finish_reason,
            "num_tokens": num_tokens,
        }
        results.append(record)

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Done! {len(results)} results saved to: {output_path}")
    print(f"  Total time: {total_elapsed:.2f}s")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    exit(main())
