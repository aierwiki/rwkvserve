"""
Command-line interface for RWKV model inference
"""

import argparse
import sys
import os

os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "1")

from rwkvserve.inference import InferencePipeline, TextGenerator


def main():
    """Entry point for rwkv-infer command"""
    parser = argparse.ArgumentParser(description="RWKV Model Inference")
    
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt (if not provided, enter interactive mode)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device (cuda/cpu)")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Precision")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature (higher = more random)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus sampling)")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 = disabled)")
    parser.add_argument("--preset", type=str, default="balanced", 
                       choices=["greedy", "creative", "precise", "balanced"],
                       help="Generation preset")
    parser.add_argument("--stream", action="store_true", help="Stream output")
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode")
    parser.add_argument("--system-prompt", type=str, default=None, help="System prompt for chat mode")
    
    args = parser.parse_args()
    
    if args.device == "cuda":
        os.environ["RWKV_CUDA_ON"] = "1"
        os.environ["RWKV_DEVICE"] = "cuda"
    else:
        os.environ["RWKV_CUDA_ON"] = "0"
        os.environ["RWKV_DEVICE"] = "cpu"
    
    print("=" * 60)
    print("RWKVServe Inference")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Precision: {args.precision}")
    print(f"Preset: {args.preset}")
    print("=" * 60)
    
    try:
        pipeline = InferencePipeline(
            model_path=args.model,
            device=args.device,
            precision=args.precision,
        )
        generator = TextGenerator(pipeline)
    except Exception as e:
        print(f"\nError: failed to load model")
        print(f"Details: {e}")
        return 1

    if args.chat:
        generator.interactive_chat(system_prompt=args.system_prompt)
        return 0

    if args.prompt is None:
        print("\nError: please provide --prompt or use --chat for interactive mode")
        return 1

    print(f"\nInput: {args.prompt}")
    print("\nOutput: ", end="")
    
    try:
        output = pipeline.generate(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stream=args.stream,
        )
        
        if not args.stream:
            print(output)
    except Exception as e:
        print(f"\n\nError: generation failed")
        print(f"Details: {e}")
        return 1

    print("\n" + "=" * 60)
    print("Generation complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
