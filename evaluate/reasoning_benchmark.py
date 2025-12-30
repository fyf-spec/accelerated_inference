# Standalone Reasoning Benchmark for Pythia-2.8B with H2O Cache
# This script evaluates reasoning tasks WITHOUT lm-eval dependency

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Import accelerated_inference modules
from accelerated_inference.kvpress.presses.benchmark_presses import StartRecentKVCache
from accelerated_inference.utils import (
    enable_gpt_neox_pos_shift_attention,
    H2OKVCache,
    LazyH2OKVCache,
)


@dataclass
class CacheConfig:
    """KV Cache configuration following H2O paper specifications"""
    mode: str  # 'baseline', 'h2o', 'lazy_h2o'
    start_size: int = 4
    heavy_ratio: float = 0.10
    recent_ratio: float = 0.10
    update_interval: int = 10
    
    def get_sizes(self, context_length: int = 2048):
        heavy_size = int(context_length * self.heavy_ratio)
        recent_size = int(context_length * self.recent_ratio)
        total_budget = self.start_size + heavy_size + recent_size
        return {
            'start_size': self.start_size,
            'heavy_size': heavy_size,
            'recent_size': recent_size,
            'total_budget': total_budget,
            'budget_ratio': total_budget / context_length
        }


class ReasoningEvaluator:
    """Standalone evaluator for reasoning benchmarks with KV cache control"""
    
    def __init__(self, model_name: str, cache_config: CacheConfig, device: str = "cuda"):
        self.model_name = model_name
        self.cache_config = cache_config
        self.device = device
        
        print(f"Loading {model_name} with cache mode: {cache_config.mode}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0
        
        self.model.eval()
        self.kv_cache = self._setup_cache()
        print(f"✓ Model loaded successfully")
    
    def _setup_cache(self):
        mode = self.cache_config.mode
        sizes = self.cache_config.get_sizes()
        
        if mode == "baseline":
            print("  Cache: Baseline (no eviction)")
            return None
        
        k_seq_dim = v_seq_dim = 2
        enable_gpt_neox_pos_shift_attention(self.model)
        
        if mode == "h2o":
            kv_cache = H2OKVCache(
                start_size=sizes['start_size'],
                recent_size=sizes['recent_size'],
                heavy_size=sizes['heavy_size'],
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
            )
            print(f"  Cache: H2O (budget={sizes['total_budget']}, {sizes['budget_ratio']:.1%})")
        elif mode == "lazy_h2o":
            kv_cache = LazyH2OKVCache(
                start_size=sizes['start_size'],
                recent_size=sizes['recent_size'],
                heavy_size=sizes['heavy_size'],
                update_interval=self.cache_config.update_interval,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
            )
            print(f"  Cache: Lazy H2O (budget={sizes['total_budget']}, interval={self.cache_config.update_interval})")
        else:
            raise ValueError(f"Unknown cache mode: {mode}")
        
        return kv_cache
    
    def _reset_cache(self):
        if self.kv_cache is None:
            return
        if hasattr(self.kv_cache, 'reset'):
            self.kv_cache.reset()
        if hasattr(self.kv_cache, 'accumulated_scores'):
            self.kv_cache.accumulated_scores = {} if isinstance(self.kv_cache.accumulated_scores, dict) else None

    def compute_loglikelihood(self, context: str, continuation: str) -> float:
        """Compute log-likelihood of continuation given context"""
        self._reset_cache()
        
        context_ids = self.tokenizer.encode(context, add_special_tokens=True)
        continuation_ids = self.tokenizer.encode(continuation, add_special_tokens=False)
        full_ids = context_ids + continuation_ids
        
        input_ids = torch.tensor([full_ids], device=self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                use_cache=True,
                output_attentions=(self.cache_config.mode in ["h2o", "lazy_h2o"]),
            )
            
            logits = outputs.logits
            continuation_start = len(context_ids) - 1
            continuation_logits = logits[0, continuation_start:continuation_start + len(continuation_ids)]
            continuation_tokens = input_ids[0, continuation_start + 1:continuation_start + 1 + len(continuation_ids)]
            
            log_probs = torch.nn.functional.log_softmax(continuation_logits, dim=-1)
            token_log_probs = log_probs[range(len(continuation_tokens)), continuation_tokens]
            
            return token_log_probs.sum().item()

    def evaluate_arc(self, split: str = "challenge", limit: Optional[int] = None) -> Dict:
        """Evaluate on ARC dataset"""
        dataset_name = f"ai2_arc" 
        config = "ARC-Challenge" if split == "challenge" else "ARC-Easy"
        
        print(f"\n📊 Evaluating ARC-{split.capitalize()}...")
        dataset = load_dataset(dataset_name, config, split="test")
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        correct = 0
        total = 0
        
        for item in tqdm(dataset, desc="ARC"):
            question = item["question"]
            choices = item["choices"]["text"]
            labels = item["choices"]["label"]
            answer_key = item["answerKey"]
            
            # Compute log-likelihood for each choice
            scores = []
            for choice in choices:
                prompt = f"Question: {question}\nAnswer: {choice}"
                # Use context as question, continuation as " {choice}"
                context = f"Question: {question}\nAnswer:"
                continuation = f" {choice}"
                score = self.compute_loglikelihood(context, continuation)
                scores.append(score)
            
            # Get prediction
            pred_idx = np.argmax(scores)
            pred_label = labels[pred_idx]
            
            if pred_label == answer_key:
                correct += 1
            total += 1
        
        acc = correct / total if total > 0 else 0
        return {"acc": acc, "correct": correct, "total": total}

    def evaluate_hellaswag(self, limit: Optional[int] = None) -> Dict:
        """Evaluate on HellaSwag dataset"""
        print(f"\n📊 Evaluating HellaSwag...")
        dataset = load_dataset("Rowan/hellaswag", split="validation")
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        correct = 0
        total = 0
        
        for item in tqdm(dataset, desc="HellaSwag"):
            context = item["ctx"]
            endings = item["endings"]
            label = int(item["label"])
            
            scores = []
            for ending in endings:
                score = self.compute_loglikelihood(context, ending)
                scores.append(score)
            
            pred = np.argmax(scores)
            if pred == label:
                correct += 1
            total += 1
        
        acc = correct / total if total > 0 else 0
        return {"acc": acc, "correct": correct, "total": total}

    def evaluate_winogrande(self, limit: Optional[int] = None) -> Dict:
        """Evaluate on Winogrande dataset"""
        print(f"\n📊 Evaluating Winogrande...")
        dataset = load_dataset("winogrande", "winogrande_xl", split="validation")
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        correct = 0
        total = 0
        
        for item in tqdm(dataset, desc="Winogrande"):
            sentence = item["sentence"]
            option1 = item["option1"]
            option2 = item["option2"]
            answer = item["answer"]
            
            # Replace underscore with each option
            sent1 = sentence.replace("_", option1)
            sent2 = sentence.replace("_", option2)
            
            # Compute log-likelihood
            score1 = self.compute_loglikelihood("", sent1)
            score2 = self.compute_loglikelihood("", sent2)
            
            pred = "1" if score1 > score2 else "2"
            if pred == answer:
                correct += 1
            total += 1
        
        acc = correct / total if total > 0 else 0
        return {"acc": acc, "correct": correct, "total": total}


def run_benchmark(model_path: str, output_dir: str, limit: Optional[int] = None):
    """Run complete benchmark with all cache strategies"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache_configs = {
        # 'baseline': CacheConfig(mode='baseline'),  # Commented out - skip baseline
        'h2o': CacheConfig(mode='h2o'),
        'lazy_h2o': CacheConfig(mode='lazy_h2o', update_interval=10),
    }
    
    all_results = {}
    
    for cache_name, cache_config in cache_configs.items():
        print(f"\n{'='*70}")
        print(f"Evaluating: {cache_name.upper()}")
        print(f"{'='*70}")
        
        evaluator = ReasoningEvaluator(model_path, cache_config)
        
        results = {}
        
        # Evaluate each task
        try:
            results['arc_challenge'] = evaluator.evaluate_arc("challenge", limit)
            print(f"  ARC-Challenge Accuracy: {results['arc_challenge']['acc']:.4f}")
        except Exception as e:
            print(f"  ARC-Challenge Error: {e}")
            results['arc_challenge'] = {"error": str(e)}
        
        try:
            results['arc_easy'] = evaluator.evaluate_arc("easy", limit)
            print(f"  ARC-Easy Accuracy: {results['arc_easy']['acc']:.4f}")
        except Exception as e:
            print(f"  ARC-Easy Error: {e}")
            results['arc_easy'] = {"error": str(e)}
        
        try:
            results['hellaswag'] = evaluator.evaluate_hellaswag(limit)
            print(f"  HellaSwag Accuracy: {results['hellaswag']['acc']:.4f}")
        except Exception as e:
            print(f"  HellaSwag Error: {e}")
            results['hellaswag'] = {"error": str(e)}
        
        try:
            results['winogrande'] = evaluator.evaluate_winogrande(limit)
            print(f"  Winogrande Accuracy: {results['winogrande']['acc']:.4f}")
        except Exception as e:
            print(f"  Winogrande Error: {e}")
            results['winogrande'] = {"error": str(e)}
        
        # Save results
        output_file = output_dir / f"results_{cache_name}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'cache_config': cache_config.get_sizes(),
                'mode': cache_config.mode,
                'results': results,
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        all_results[cache_name] = results
        
        # Cleanup
        del evaluator
        torch.cuda.empty_cache()
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="H2O Reasoning Benchmark")
    parser.add_argument("--model", type=str, default="e:/project/accelerated_inference/pythia-2.8b-local", 
                       help="Model path or HuggingFace name")
    parser.add_argument("--output", type=str, default="../outputs/reasoning_benchmark",
                       help="Output directory")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of samples per task (for testing)")
    
    args = parser.parse_args()
    
    results = run_benchmark(args.model, args.output, args.limit)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for cache_name, cache_results in results.items():
        print(f"\n{cache_name.upper()}:")
        for task, metrics in cache_results.items():
            if 'acc' in metrics:
                print(f"  {task}: {metrics['acc']:.4f} ({metrics['acc']*100:.2f}%)")
            else:
                print(f"  {task}: Error")
