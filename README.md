# Accelerated LLM Inference with KV Cache Compression

This repository implements and benchmarks **KV Cache compression strategies** for accelerated Large Language Model inference, focusing on the **Pythia-2.8B** model.

## Supported Methods

| Method | Description |
|--------|-------------|
| **Baseline** | Full KV cache (no eviction) |
| **StreamingLLM** | Sink + Recent window (O(1) eviction) |
| **H2O** | Heavy Hitter Oracle - attention-based eviction |
| **LazyH2O** | Periodic H2O (lazy eviction every N steps) |
| **SepLLM** | Separator-aware token retention |
| **Unified** | Combines Initial + Separator + Heavy Hitter + Local |

---

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/accelerated_inference.git
cd accelerated_inference

# Create conda environment (recommended)
conda create -n accel_infer python=3.10
conda activate accel_infer

# Install dependencies
pip install torch>=2.0.0 transformers==4.33.0 numpy tqdm
pip install matplotlib datasets huggingface_hub  # for benchmarking

# Install the package
pip install -e .
```

### 2. Download Model

```bash
python download_model.py --output_dir pythia-2.8b-local
```

This downloads **EleutherAI/pythia-2.8b** (~5.6GB) to `pythia-2.8b-local/`.

### 3. Prepare Dataset (PG-19)

The `dataset/pg19/` directory should contain `.txt` files from the PG-19 dataset. If you need to download them:

```python
from datasets import load_dataset
import os

ds = load_dataset("pg19", split="test")
os.makedirs("dataset/pg19", exist_ok=True)
for i, sample in enumerate(ds):
    if i >= 100:  # Download first 100 books
        break
    with open(f"dataset/pg19/book_{i:03d}.txt", "w", encoding="utf-8") as f:
        f.write(sample["text"])
```

---

## Benchmark: Perplexity (PG-19)

Evaluate perplexity on the PG-19 dataset with different KV cache strategies.

### Run Benchmarks

```bash
# Baseline (chunked evaluation, 512 tokens per chunk)
python evaluate/eval_ppl_pg19.py --mode baseline --num_samples 10

# StreamingLLM
python evaluate/eval_ppl_pg19.py --mode streaming --start_size 4 --recent_size 252 --num_samples 10

# H2O
python evaluate/eval_ppl_pg19.py --mode h2o --start_size 4 --recent_size 128 --heavy_size 128 --num_samples 10

# LazyH2O
python evaluate/eval_ppl_pg19.py --mode lazy_h2o --start_size 4 --recent_size 128 --heavy_size 128 --update_interval 10 --num_samples 10
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `baseline` | KV cache strategy: `baseline`, `streaming`, `h2o`, `lazy_h2o`, `sepllm`, `unified` |
| `--model_name_or_path` | `pythia-2.8b-local` | Path to the model |
| `--data_dir` | `dataset/pg19` | Directory containing PG-19 `.txt` files |
| `--num_samples` | `10` | Number of text files to evaluate |
| `--num_eval_tokens` | `None` | Maximum tokens to evaluate (None = all) |
| `--start_size` | `4` | Number of sink tokens |
| `--recent_size` | `252` | Number of recent tokens |
| `--heavy_size` | `128` | Number of Heavy Hitter tokens (H2O only) |
| `--update_interval` | `10` | H2O update interval (LazyH2O only) |

### Output

Results are saved to `outputs/ppl_pg19/ppl_results_{mode}.json`:
```json
{
  "mode": "streaming",
  "ppl": 10.234,
  "num_tokens": 50000,
  "num_samples": 10,
  "config": {
    "start_size": 4,
    "recent_size": 252
  }
}
```

---

## Benchmark: Decode Speed

Measure decode latency at different sequence lengths.

### Run Benchmarks

```bash
# Baseline
python evaluate/eval_speed_benchmark.py --mode baseline --seq_lengths 256 512 1024 2048

# StreamingLLM
python evaluate/eval_speed_benchmark.py --mode streaming --seq_lengths 256 512 1024 2048 4096

# H2O
python evaluate/eval_speed_benchmark.py --mode h2o --seq_lengths 256 512 1024 2048 4096

# LazyH2O
python evaluate/eval_speed_benchmark.py --mode lazy_h2o --update_interval 10 --seq_lengths 256 512 1024 2048 4096
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `baseline` | KV cache strategy |
| `--seq_lengths` | `[256, 512, 1024, 2048]` | Sequence lengths to test |
| `--num_decode_tokens` | `128` | Number of decode tokens to measure |
| `--output_dir` | `outputs/speed_benchmark` | Output directory |

### Output

Results are saved to `outputs/speed_benchmark/speed_results_{mode}.json` and a plot is generated as `speed_benchmark_{mode}.png`:

```json
{
  "config": { ... },
  "results": [
    {
      "seq_length": 1024,
      "kv_cache_size": 256,
      "avg_decode_latency_ms": 12.5,
      "tokens_per_sec": 80.0
    }
  ]
}
```

---

## Benchmark Results
 tested on RTX 4090
### Perplexity on PG-19 (Pythia-2.8B)

Evaluation on 1 PG-19 book (~59k tokens), cache size = 256 tokens (start=4, recent=252), heavy=128 for H2O methods.

| Method | PPL ↓ | Cache Size |
|--------|-------|------------|
| Baseline (512-chunk) | 10.52 | N/A (full) |
| StreamingLLM | 9.58 | 256 |
| H2O | 9.53 | 384 |
| **LazyH2O (k=10)** | **9.45** | 384 |
| SepLLM | 9.56 | ~324 |
| Unified | 9.46 | ~452 |

**Key Finding**: All KV cache compression methods achieve **lower PPL** than the chunked baseline while using only 256-384 tokens of cache, demonstrating that streaming evaluation with proper cache management outperforms fixed-window chunked evaluation.

---

### Decode Latency (Pythia-2.8B)

Average decode latency at different sequence lengths (128 decode tokens measured).

| Seq Length | Baseline | StreamingLLM | H2O | LazyH2O |
|------------|----------|--------------|-----|---------|
| 512 | 28.2 ms | 32.7 ms | 33.1 ms | 35.2 ms |
| 1024 | 27.9 ms | 32.9 ms | 32.7 ms | 34.8 ms |
| 2048 | 63.9 ms | 32.6 ms | 32.8 ms | 34.8 ms |
| 4096 | 162.5 ms | - | - | - |

| Method | KV Cache Size | Latency Scaling |
|--------|---------------|-----------------|
| Baseline | Grows with seq | O(n) - **degrades** |
| StreamingLLM | Fixed (256) | O(1) - **constant** |
| H2O | Fixed (384) | O(1) - **constant** |
| LazyH2O | Fixed (384) | O(1) - **constant** |

**Key Finding**: 
- **Baseline** latency grows from 28ms → 163ms as sequence length increases (5.8× degradation)
- **StreamingLLM/H2O/LazyH2O** maintain constant ~33ms latency regardless of sequence length
- At 4096 tokens, KV cache compression provides **~5× speedup**

---

## Project Structure

```
accelerated_inference/
├── accelerated_inference/           # Core library
│   ├── kvpress/                     # KV Cache compression implementations
│   │   ├── base_press.py            # BasePress abstract class
│   │   └── presses/
│   │       ├── benchmark_presses.py # StreamLLM, H2O, SepLLM caches
│   │       └── unified_press.py     # Unified/LazyUnified caches
│   └── utils.py                     # Position shift attention, H2OKVCache, LazyH2OKVCache
├── evaluate/
│   ├── eval_ppl_pg19.py             # Perplexity evaluation script
│   └── eval_speed_benchmark.py      # Speed/latency benchmark script
├── dataset/
│   └── pg19/                        # PG-19 text files (.txt)
├── outputs/                         # Benchmark results and plots
├── download_model.py                # Download Pythia-2.8B model
├── setup.py                         # Package installation
└── README.md
```

---

## Hardware Requirements

- **GPU**: NVIDIA GPU with at least 12GB VRAM (tested on RTX 3090/4090)
- **Memory**: 16GB+ RAM recommended
- **Disk**: ~10GB for model and dataset


## License

MIT License
