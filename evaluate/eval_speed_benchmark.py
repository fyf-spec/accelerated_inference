"""
Speed/Latency Benchmark for Pythia-2.8b with StreamingLLM

This script measures the DECODE latency at specific sequence lengths,
matching the StreamingLLM paper's efficiency comparison figure.

Test points: 256, 512, 1024, 2048, 4096

Usage:
    # Baseline (no streaming)
    python evaluate/eval_speed_benchmark.py \
        --model_name_or_path pythia-2.8b-local \
        --output_dir outputs/speed_baseline

    # With StreamingLLM
    python evaluate/eval_speed_benchmark.py \
        --model_name_or_path pythia-2.8b-local \
        --enable_start_recent_kv_cache \
        --enable_pos_shift \
        --start_size 4 \
        --recent_size 252 \
        --output_dir outputs/speed_streaming
"""

import torch
import time
import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import from accelerated_inference
from accelerated_inference.kvpress.presses.benchmark_presses import StartRecentKVCache
from accelerated_inference.utils import enable_gpt_neox_pos_shift_attention

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_args():
    parser = argparse.ArgumentParser(description="Speed benchmark for Pythia with StreamingLLM")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="pythia-2.8b-local",
        help="Path to the model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/speed_benchmark",
        help="Output directory for results"
    )
    
    # Sequence length settings - matching StreamingLLM paper Figure
    parser.add_argument(
        "--seq_lengths",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048, 4096],
        help="Sequence lengths to benchmark (matching StreamingLLM paper)"
    )
    parser.add_argument(
        "--num_decode_tokens",
        type=int,
        default=128,
        help="Number of decode tokens to measure average latency"
    )
    
    # StreamingLLM settings
    parser.add_argument("--enable_start_recent_kv_cache", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=252)
    parser.add_argument("--enable_pos_shift", action="store_true")
    
    return parser.parse_args()


def load_model(model_name_or_path):
    """Load model and tokenizer"""
    print(f"Loading model from {model_name_or_path} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0
    
    model.eval()
    return model, tokenizer


def setup_streaming(model, args):
    """Setup StreamingLLM components"""
    kv_cache = None
    
    if args.enable_start_recent_kv_cache:
        # For Pythia (GPT-NeoX architecture)
        k_seq_dim = v_seq_dim = 2
        
        kv_cache = StartRecentKVCache(
            start_size=args.start_size,
            recent_size=args.recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
        )
        print(f"KV Cache enabled: start_size={args.start_size}, recent_size={args.recent_size}")
    
    if args.enable_pos_shift:
        enable_gpt_neox_pos_shift_attention(model)
        print("Position shift attention enabled")
    
    return kv_cache


def generate_random_input(tokenizer, total_tokens, device="cuda"):
    """Generate random input tokens for benchmarking"""
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(
        low=100,
        high=vocab_size - 100,
        size=(1, total_tokens),
        dtype=torch.long,
        device=device
    )
    return input_ids


def benchmark_at_seq_length(model, tokenizer, seq_length, num_decode_tokens, kv_cache, device="cuda"):
    """
    Benchmark decode latency at a specific sequence length.
    
    Process:
    1. Prefill with (seq_length - 1) tokens
    2. Decode num_decode_tokens more tokens, measuring latency
    3. Return average decode latency
    """
    # Generate random input
    total_tokens = seq_length + num_decode_tokens
    input_ids = generate_random_input(tokenizer, total_tokens, device)
    
    past_key_values = None
    
    # Prefill phase: process (seq_length - 1) tokens at once
    prefill_size = seq_length - 1
    prefill_ids = input_ids[:, :prefill_size]
    
    with torch.no_grad():
        outputs = model(
            prefill_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        if kv_cache is not None:
            past_key_values = kv_cache(past_key_values)
    
    # Now decode from position (seq_length - 1) onwards
    # First, decode one token to reach exactly seq_length
    next_token = input_ids[:, prefill_size:prefill_size+1]
    with torch.no_grad():
        outputs = model(
            next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        if kv_cache is not None:
            past_key_values = kv_cache(past_key_values)
    
    # Now we're at exactly seq_length tokens
    # Measure decode latency for num_decode_tokens more tokens
    decode_times = []
    current_pos = seq_length
    
    for i in range(num_decode_tokens):
        next_token = input_ids[:, current_pos:current_pos+1]
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = model(
                next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            if kv_cache is not None:
                past_key_values = kv_cache(past_key_values)
        
        torch.cuda.synchronize()
        decode_time = time.perf_counter() - start_time
        decode_times.append(decode_time)
        
        current_pos += 1
    
    # Get KV cache size
    if past_key_values is not None:
        kv_size = past_key_values[0][0].size(2)
    else:
        kv_size = current_pos
    
    # Calculate statistics
    avg_latency_ms = float(np.mean(decode_times)) * 1000
    std_latency_ms = float(np.std(decode_times)) * 1000
    total_latency_ms = float(np.sum(decode_times)) * 1000
    
    return {
        "seq_length": int(seq_length),
        "kv_cache_size": int(kv_size),
        "avg_decode_latency_ms": avg_latency_ms,
        "std_decode_latency_ms": std_latency_ms,
        "total_decode_latency_ms": total_latency_ms,
        "tokens_per_sec": 1000 / avg_latency_ms if avg_latency_ms > 0 else 0,
    }


def plot_results(results, output_dir, is_streaming, args):
    """Plot results matching StreamingLLM paper style"""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot generation")
        return
    
    mode_str = "StreamingLLM" if is_streaming else "Baseline"
    color = '#8B0000' if is_streaming else '#808080'  # Dark red for streaming, gray for baseline
    
    seq_lengths = [r["seq_length"] for r in results]
    latencies = [r["avg_decode_latency_ms"] for r in results]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Bar chart like the paper
    x = np.arange(len(seq_lengths))
    bars = ax.bar(x, latencies, color=color, width=0.6, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, latency in zip(bars, latencies):
        height = bar.get_height()
        ax.annotate(f'{latency:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title(f'Decode Latency - {mode_str}\n(Pythia-2.8B)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lengths)
    ax.set_ylim(0, max(latencies) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "speed_benchmark.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.close()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model(args.model_name_or_path)
    
    # Setup StreamingLLM if enabled
    kv_cache = setup_streaming(model, args)
    is_streaming = args.enable_start_recent_kv_cache
    
    print(f"\n{'='*60}")
    print(f"Speed Benchmark Configuration")
    print(f"{'='*60}")
    print(f"Model: {args.model_name_or_path}")
    print(f"Mode: {'StreamingLLM' if is_streaming else 'Baseline'}")
    if is_streaming:
        print(f"  - start_size: {args.start_size}")
        print(f"  - recent_size: {args.recent_size}")
        print(f"  - cache_size: {args.start_size + args.recent_size}")
        print(f"  - pos_shift: {args.enable_pos_shift}")
    print(f"Sequence lengths to test: {args.seq_lengths}")
    print(f"Decode tokens per test: {args.num_decode_tokens}")
    print(f"{'='*60}\n")
    
    # Run benchmarks for each sequence length
    results = []
    
    for seq_length in tqdm(args.seq_lengths, desc="Benchmarking"):
        print(f"\nTesting sequence length: {seq_length}")
        
        try:
            result = benchmark_at_seq_length(
                model, tokenizer, seq_length, 
                args.num_decode_tokens, kv_cache
            )
            results.append(result)
            
            print(f"  KV Cache Size: {result['kv_cache_size']}")
            print(f"  Avg Decode Latency: {result['avg_decode_latency_ms']:.2f} ± {result['std_decode_latency_ms']:.2f} ms")
            print(f"  Throughput: {result['tokens_per_sec']:.1f} tokens/sec")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [OOM] Skipping seq_length={seq_length}")
                torch.cuda.empty_cache()
                continue
            else:
                raise
        
        # Clear cache between tests
        torch.cuda.empty_cache()
    
    # Save results
    config_dict = {k: v for k, v in vars(args).items()}
    
    results_path = os.path.join(args.output_dir, "speed_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "config": config_dict,
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Plot results
    plot_results(results, args.output_dir, is_streaming, args)
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Seq Length':>12} {'KV Cache':>12} {'Latency (ms)':>15} {'Tokens/sec':>15}")
    print(f"{'='*80}")
    for r in results:
        print(f"{r['seq_length']:>12} {r['kv_cache_size']:>12} {r['avg_decode_latency_ms']:>15.2f} {r['tokens_per_sec']:>15.1f}")
    print(f"{'='*80}")
    
    # Summary
    print(f"\n📊 Summary:")
    if is_streaming:
        print(f"   StreamingLLM maintains constant KV cache size of {args.start_size + args.recent_size}")
        print(f"   Decode latency should be relatively CONSTANT across all sequence lengths")
    else:
        print(f"   Baseline KV cache grows with sequence length")
        print(f"   Decode latency INCREASES as sequence length grows")


if __name__ == "__main__":
    main()
