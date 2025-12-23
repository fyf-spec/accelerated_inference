"""
Batch PPL Testing Script

This script runs PPL evaluation for all 6 methods in eval_ppl_pg19.py
and records the output results to a summary file.

Methods tested:
1. baseline - Full KV cache (chunked evaluation)
2. streaming - StreamingLLM (Sink + Recent window)
3. h2o - Heavy Hitter + Recent
4. lazy_h2o - Periodic H2O
5. sepllm - SepLLM cache
6. unified - Unified KV cache

Usage:
    python evaluate/batch_ppl_test.py
    python evaluate/batch_ppl_test.py --num_samples 5 --num_eval_tokens 1000
"""

import subprocess
import sys
import os
import json
import time
import signal
from datetime import datetime

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "eval_ppl_pg19.py")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "ppl_pg19_batch")

# Default parameters for all methods
DEFAULT_PARAMS = {
    "model_name_or_path": "pythia-2.8b-local",
    "data_dir": "dataset/pg19",
    "num_samples": 1,
    "num_eval_tokens": None,  # Set to a number for quick testing, e.g., 1000
    
    # Cache parameters
    "start_size": 4,
    "recent_size": 252,
    "heavy_size": 128,
    "update_interval": 10,
    "separator_size": 64,
    "local_size": 256,
    "context_length": 512,
}

# Method configurations
METHODS = [
    {
        "name": "baseline",
        "description": "Full KV cache (chunked evaluation, 512 tokens per chunk)",
        "extra_args": ["--context_length", str(DEFAULT_PARAMS["context_length"])],
    },
    {
        "name": "streaming",
        "description": "StreamingLLM (Sink + Recent window)",
        "extra_args": [],
    },
    {
        "name": "h2o",
        "description": "Heavy Hitter + Recent (attention-based eviction)",
        "extra_args": ["--heavy_size", str(DEFAULT_PARAMS["heavy_size"])],
    },
    {
        "name": "lazy_h2o",
        "description": "Periodic H2O (lazy eviction)",
        "extra_args": [
            "--heavy_size", str(DEFAULT_PARAMS["heavy_size"]),
            "--update_interval", str(DEFAULT_PARAMS["update_interval"]),
        ],
    },
    {
        "name": "sepllm",
        "description": "SepLLM cache (separator-based)",
        "extra_args": [
            "--separator_size", str(DEFAULT_PARAMS["separator_size"]),
            "--local_size", str(DEFAULT_PARAMS["local_size"]),
        ],
    },
    {
        "name": "unified",
        "description": "Unified KV cache (combines all strategies)",
        "extra_args": [
            "--separator_size", str(DEFAULT_PARAMS["separator_size"]),
            "--heavy_size", str(DEFAULT_PARAMS["heavy_size"]),
            "--local_size", str(DEFAULT_PARAMS["local_size"]),
        ],
    },
]


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Batch PPL testing for all 6 methods")
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_PARAMS["model_name_or_path"])
    parser.add_argument("--data_dir", type=str, default=DEFAULT_PARAMS["data_dir"])
    parser.add_argument("--num_samples", type=int, default=DEFAULT_PARAMS["num_samples"])
    parser.add_argument("--num_eval_tokens", type=int, default=DEFAULT_PARAMS["num_eval_tokens"],
                        help="Max tokens to evaluate per method (None = all)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--methods", type=str, nargs="+", 
                        default=["baseline", "streaming", "h2o", "lazy_h2o", "sepllm", "unified"],
                        help="Methods to test (default: all 6)")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    return parser.parse_args()


def run_method(method_config, args):
    """Run a single method and return the result."""
    mode = method_config["name"]
    print(f"\n{'='*70}")
    print(f"Testing: {mode.upper()}")
    print(f"Description: {method_config['description']}")
    print(f"{'='*70}")
    
    # Build command
    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--mode", mode,
        "--model_name_or_path", args.model_name_or_path,
        "--data_dir", args.data_dir,
        "--num_samples", str(args.num_samples),
        "--output_dir", args.output_dir,
        "--start_size", str(DEFAULT_PARAMS["start_size"]),
        "--recent_size", str(DEFAULT_PARAMS["recent_size"]),
    ]
    
    if args.num_eval_tokens is not None:
        cmd.extend(["--num_eval_tokens", str(args.num_eval_tokens)])
    
    # Add method-specific arguments
    cmd.extend(method_config["extra_args"])
    
    print(f"Command: {' '.join(cmd)}\n")
    
    if args.dry_run:
        return {
            "mode": mode,
            "status": "dry_run",
            "ppl": None,
            "time_seconds": 0,
        }
    
    # Run the command
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3600 * 2,  # 2 hour timeout
        )
        elapsed_time = time.time() - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Try to load the result file
        result_file = os.path.join(args.output_dir, f"ppl_results_{mode}.json")
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                result_data = json.load(f)
            return {
                "mode": mode,
                "status": "success",
                "ppl": result_data.get("ppl"),
                "num_tokens": result_data.get("num_tokens"),
                "time_seconds": elapsed_time,
                "result_file": result_file,
            }
        else:
            return {
                "mode": mode,
                "status": "error",
                "error": "Result file not found",
                "time_seconds": elapsed_time,
                "stdout": result.stdout[-1000:] if result.stdout else "",
                "stderr": result.stderr[-1000:] if result.stderr else "",
            }
            
    except subprocess.TimeoutExpired:
        return {
            "mode": mode,
            "status": "timeout",
            "time_seconds": time.time() - start_time,
        }
    except Exception as e:
        return {
            "mode": mode,
            "status": "exception",
            "error": str(e),
            "time_seconds": time.time() - start_time,
        }


def save_progress(args, results, total_start_time, is_final=False, interrupted=False):
    """Save current progress to files. Called after each method and at the end."""
    total_time = time.time() - total_start_time
    
    status_str = "completed" if is_final else ("interrupted" if interrupted else "in_progress")
    
    # Build summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "status": status_str,
        "config": {
            "model_name_or_path": args.model_name_or_path,
            "data_dir": args.data_dir,
            "num_samples": args.num_samples,
            "num_eval_tokens": args.num_eval_tokens,
            "default_params": DEFAULT_PARAMS,
        },
        "results": results,
        "total_time_seconds": total_time,
        "methods_completed": len(results),
    }
    
    # Save progress JSON (always overwrite with latest)
    progress_file = os.path.join(args.output_dir, "batch_ppl_progress.json")
    with open(progress_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    # If final, also save to summary file
    if is_final or interrupted:
        summary_file = os.path.join(args.output_dir, "batch_ppl_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save readable text report
        report_file = os.path.join(args.output_dir, "batch_ppl_report.txt")
        with open(report_file, "w") as f:
            f.write("="*70 + "\n")
            f.write("Batch PPL Testing Report\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if interrupted:
                f.write("*** INTERRUPTED - Partial Results ***\n")
            f.write("="*70 + "\n\n")
            
            f.write("Configuration:\n")
            f.write(f"  Model: {args.model_name_or_path}\n")
            f.write(f"  Data directory: {args.data_dir}\n")
            f.write(f"  Num samples: {args.num_samples}\n")
            f.write(f"  Num eval tokens: {args.num_eval_tokens if args.num_eval_tokens else 'All'}\n")
            f.write(f"  Cache params: start={DEFAULT_PARAMS['start_size']}, recent={DEFAULT_PARAMS['recent_size']}\n")
            f.write("\n")
            
            f.write("Results:\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Method':<15} {'Status':<10} {'PPL':<15} {'Tokens':<12} {'Time (s)':<10}\n")
            f.write("-"*70 + "\n")
            
            for r in results:
                ppl_str = f"{r['ppl']:.4f}" if r.get('ppl') else "N/A"
                tokens_str = str(r.get('num_tokens', 'N/A'))
                time_str = f"{r['time_seconds']:.1f}"
                f.write(f"{r['mode']:<15} {r['status']:<10} {ppl_str:<15} {tokens_str:<12} {time_str:<10}\n")
            
            f.write("-"*70 + "\n")
            f.write(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)\n")
            f.write("="*70 + "\n")
    
    return total_time


def print_summary(results, total_time):
    """Print summary table to console."""
    print("\n" + "="*70)
    print("SUMMARY RESULTS")
    print("="*70)
    print(f"{'Method':<15} {'Status':<10} {'PPL':<15} {'Tokens':<12} {'Time (s)':<10}")
    print("-"*70)
    
    for r in results:
        ppl_str = f"{r['ppl']:.4f}" if r.get('ppl') else "N/A"
        tokens_str = str(r.get('num_tokens', 'N/A'))
        time_str = f"{r['time_seconds']:.1f}"
        status = r['status']
        print(f"{r['mode']:<15} {status:<10} {ppl_str:<15} {tokens_str:<12} {time_str:<10}")
    
    print("-"*70)
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print("="*70)


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("="*70)
    print("Batch PPL Testing Script")
    print("="*70)
    print(f"Model: {args.model_name_or_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Num samples: {args.num_samples}")
    print(f"Num eval tokens: {args.num_eval_tokens if args.num_eval_tokens else 'All'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Methods to test: {args.methods}")
    print(f"Dry run: {args.dry_run}")
    print("="*70)
    
    # Filter methods based on user selection
    selected_methods = [m for m in METHODS if m["name"] in args.methods]
    
    if not selected_methods:
        print("No valid methods selected!")
        return
    
    # Track results and timing
    results = []
    total_start_time = time.time()
    interrupted = False
    
    # Setup signal handler for graceful interruption
    def signal_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n\n*** Interrupted! Saving progress... ***")
        total_time = save_progress(args, results, total_start_time, is_final=False, interrupted=True)
        print_summary(results, total_time)
        print(f"\nProgress saved to: {os.path.join(args.output_dir, 'batch_ppl_progress.json')}")
        print(f"Summary saved to: {os.path.join(args.output_dir, 'batch_ppl_summary.json')}")
        sys.exit(1)
    
    # Register signal handler (SIGINT = Ctrl+C)
    original_handler = signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Run all methods
        for i, method in enumerate(selected_methods):
            print(f"\n[{i+1}/{len(selected_methods)}] Running method: {method['name']}")
            
            result = run_method(method, args)
            results.append(result)
            
            # Save progress after each method completes
            save_progress(args, results, total_start_time, is_final=False, interrupted=False)
            print(f"  -> Progress saved ({len(results)}/{len(selected_methods)} methods completed)")
        
        # All methods completed successfully
        total_time = save_progress(args, results, total_start_time, is_final=True, interrupted=False)
        
        # Print final summary
        print_summary(results, total_time)
        print(f"\nSummary saved to: {os.path.join(args.output_dir, 'batch_ppl_summary.json')}")
        print(f"Report saved to: {os.path.join(args.output_dir, 'batch_ppl_report.txt')}")
        
    except Exception as e:
        # Handle unexpected exceptions
        print(f"\n*** Unexpected error: {e} ***")
        print("Saving progress before exit...")
        total_time = save_progress(args, results, total_start_time, is_final=False, interrupted=True)
        print_summary(results, total_time)
        raise
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)


if __name__ == "__main__":
    main()
