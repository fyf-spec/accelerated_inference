"""
PPL Evaluation for Pythia on PG-19 Dataset (Local Files)

This script evaluates perplexity on the local PG-19 dataset files.
No online dataset loading required.

Usage:
    # Baseline (no streaming)
    python evaluate/eval_ppl_pg19.py \
        --model_name_or_path pythia-2.8b-local \
        --data_dir dataset/pg19 \
        --output_dir outputs/ppl_pg19_baseline

    # With StreamingLLM
    python evaluate/eval_ppl_pg19.py \
        --model_name_or_path pythia-2.8b-local \
        --data_dir dataset/pg19 \
        --enable_start_recent_kv_cache \
        --enable_pos_shift \
        --start_size 4 \
        --recent_size 252 \
        --output_dir outputs/ppl_pg19_streaming
"""

import torch
import os
import glob
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import CrossEntropyLoss

# Import from accelerated_inference
from accelerated_inference.kvpress.presses.benchmark_presses import StartRecentKVCache
from accelerated_inference.utils import enable_gpt_neox_pos_shift_attention

device = "cuda"


def parse_args():
    parser = argparse.ArgumentParser(description="PPL evaluation on PG-19 dataset")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="pythia-2.8b-local",
        help="Path to the model"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/pg19",
        help="Directory containing PG-19 txt files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/ppl_pg19",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of text files to evaluate"
    )
    parser.add_argument(
        "--num_eval_tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to evaluate (None = all)"
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


def load_pg19_texts(data_dir, num_samples):
    """Load text files from PG-19 dataset directory"""
    txt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    
    if len(txt_files) == 0:
        raise ValueError(f"No .txt files found in {data_dir}")
    
    print(f"Found {len(txt_files)} text files in {data_dir}")
    
    texts = []
    for txt_file in txt_files[:num_samples]:
        with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            texts.append({
                "filename": os.path.basename(txt_file),
                "text": text
            })
    
    return texts


def evaluate_ppl(model, tokenizer, texts, kv_cache, args):
    """Evaluate perplexity on the given texts"""
    nlls = []
    loss_fn = CrossEntropyLoss(reduction="none")
    num_eval_tokens = 0
    
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = open(f"{args.output_dir}/log.txt", "w")
    
    for sample_idx, sample in enumerate(texts):
        filename = sample["filename"]
        text = sample["text"]
        
        print(f"\n[{sample_idx + 1}/{len(texts)}] Processing: {filename}")
        
        # Tokenize
        encodings = tokenizer(text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)
        print(f"  Sequence length: {seq_len} tokens")
        
        # Reset KV cache for each document
        past_key_values = None
        
        # Evaluate token by token
        pbar = tqdm(range(0, seq_len - 1), desc=f"Evaluating {filename}")
        
        for idx in pbar:
            input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
            
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits.view(-1, model.config.vocab_size)
                past_key_values = outputs.past_key_values
                
                # Get label (next token)
                label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
                
                # Compute NLL
                neg_log_likelihood = loss_fn(logits, label)
                
                # Apply KV cache eviction if enabled
                if kv_cache is not None:
                    past_key_values = kv_cache(past_key_values)
            
            nlls.append(neg_log_likelihood)
            
            # Update progress bar
            current_ppl = torch.exp(torch.stack(nlls).mean()).item()
            pbar.set_description(
                f"nll: {neg_log_likelihood.item():.2f}, ppl: {current_ppl:.2f}"
            )
            
            # Log
            print(neg_log_likelihood.item(), file=log_file, flush=True)
            
            num_eval_tokens += 1
            if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
                break
        
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            print(f"Reached {args.num_eval_tokens} tokens limit, stopping evaluation")
            break
    
    log_file.close()
    
    # Compute final PPL
    final_ppl = torch.exp(torch.stack(nlls).mean()).item()
    
    return {
        "ppl": final_ppl,
        "num_tokens": num_eval_tokens,
        "num_samples": len(texts),
    }


def main():
    args = parse_args()
    
    # Print configuration
    is_streaming = args.enable_start_recent_kv_cache
    
    print(f"\n{'='*60}")
    print(f"PPL Evaluation Configuration")
    print(f"{'='*60}")
    print(f"Model: {args.model_name_or_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Mode: {'StreamingLLM' if is_streaming else 'Baseline'}")
    if is_streaming:
        print(f"  - start_size: {args.start_size}")
        print(f"  - recent_size: {args.recent_size}")
        print(f"  - pos_shift: {args.enable_pos_shift}")
    print(f"Num samples: {args.num_samples}")
    print(f"Num eval tokens: {args.num_eval_tokens if args.num_eval_tokens else 'All'}")
    print(f"{'='*60}\n")
    
    # Load model
    model, tokenizer = load_model(args.model_name_or_path)
    
    # Setup StreamingLLM
    kv_cache = setup_streaming(model, args)
    
    # Load PG-19 texts
    texts = load_pg19_texts(args.data_dir, args.num_samples)
    
    # Evaluate
    results = evaluate_ppl(model, tokenizer, texts, kv_cache, args)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(f"{args.output_dir}/ppl.txt", "w") as f:
        f.write(f"{results['ppl']}\n")
    
    with open(f"{args.output_dir}/results.txt", "w") as f:
        f.write(f"Mode: {'StreamingLLM' if is_streaming else 'Baseline'}\n")
        f.write(f"PPL: {results['ppl']:.4f}\n")
        f.write(f"Num tokens evaluated: {results['num_tokens']}\n")
        f.write(f"Num samples: {results['num_samples']}\n")
    
    # Print final results
    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"Mode: {'StreamingLLM' if is_streaming else 'Baseline'}")
    print(f"Final PPL: {results['ppl']:.4f}")
    print(f"Tokens evaluated: {results['num_tokens']}")
    print(f"Samples processed: {results['num_samples']}")
    print(f"{'='*60}")
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
