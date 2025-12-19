# Accelerated Inference Project

本项目旨在通过 KV Cache 压缩技术加速 Pythia 模型的推理过程。项目包含自定义的 `kvpress` 库用于实现各种压缩算法（Presses），以及 StreamingLLM 实现和评估脚本。

## 目录结构

- **`accelerated_inference/`**: 核心库
  - `kvpress/`: KV Cache 压缩实现
    - `base_press.py`: 压缩算法基类 `BasePress`
    - `presses/benchmark_presses.py`: 包含 `StreamLLMPress`, `SnapKVPress`, `StartRecentKVCache`
  - `utils.py`: StreamingLLM 工具函数（模型加载、GPT-NeoX 位置偏移注意力等）
- **`evaluate/`**: 评估脚本
  - `eval_ppl_pg19.py`: PG-19 数据集上的 PPL 评估
  - `eval_speed_benchmark.py`: 解码延迟基准测试
- **`dataset/`**: 数据集（Wikitext-2, PG19）
- **`pythia-2.8b-local/`**: 本地模型文件

## 环境准备

```bash
pip install torch transformers datasets tqdm matplotlib numpy
```

## 使用说明

### 1. StreamingLLM 评估

```bash
cd accelerated_inference
$env:PYTHONPATH = "."  # Windows PowerShell
# export PYTHONPATH="."  # Linux/Mac

# Baseline PPL 评估
python evaluate/eval_ppl_pg19.py \
    --model_name_or_path pythia-2.8b-local \
    --data_dir dataset/pg19 \
    --output_dir outputs/ppl_baseline

# StreamingLLM PPL 评估
python evaluate/eval_ppl_pg19.py \
    --model_name_or_path pythia-2.8b-local \
    --data_dir dataset/pg19 \
    --enable_start_recent_kv_cache \
    --enable_pos_shift \
    --start_size 4 \
    --recent_size 252 \
    --output_dir outputs/ppl_streaming
```

### 2. 速度基准测试

```bash
# Baseline 解码延迟
python evaluate/eval_speed_benchmark.py \
    --model_name_or_path pythia-2.8b-local \
    --output_dir outputs/speed_baseline

# StreamingLLM 解码延迟
python evaluate/eval_speed_benchmark.py \
    --model_name_or_path pythia-2.8b-local \
    --enable_start_recent_kv_cache \
    --enable_pos_shift \
    --output_dir outputs/speed_streaming
```

### 3. 使用 StartRecentKVCache

```python
from accelerated_inference.kvpress.presses.benchmark_presses import StartRecentKVCache
from accelerated_inference.utils import enable_gpt_neox_pos_shift_attention

# 启用位置偏移注意力
enable_gpt_neox_pos_shift_attention(model)

# 创建 KV Cache
kv_cache = StartRecentKVCache(start_size=4, recent_size=252)

# 在生成循环中使用
outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
past_key_values = kv_cache(outputs.past_key_values)  # 自动驱逐中间 token
```

### 4. 使用 KVPress 进行加速

```python
from transformers import AutoModelForCausalLM
from accelerated_inference.kvpress.presses.benchmark_presses import StreamLLMPress

model = AutoModelForCausalLM.from_pretrained("pythia-2.8b-local")
press = StreamLLMPress(compression_ratio=0.5, num_sinks=4)

with press(model):
    output = model.generate(...)
```

