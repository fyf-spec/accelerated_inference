# Instructions for Installing and Using accelerated_inference Package

## Installation

To install the accelerated_inference package in development mode:

```bash
cd e:/project/accelerated_inference
pip install -e .
```

This will install the package and make it importable from anywhere in your Python environment.

## Fixing lm-eval Compatibility Issue

The error you're seeing is because `lm-eval` version >= 0.4.0 requires newer transformers features that don't exist in transformers 0.44.3.

### Solution 1: Use Compatible lm-eval Version (Recommended)

Install an older version of lm-eval that's compatible with transformers 0.44.3:

```bash
pip install "lm-eval<0.4.0"
```

Or specifically:

```bash
pip install lm-eval==0.3.0
```

### Solution 2: Apply Compatibility Patch

Use the provided compatibility patch in your notebook. Replace Cell 2 (the imports cell) with:

```python
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

# Add project root to path
project_root = Path.cwd().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import transformers first
from transformers import AutoModelForCausalLM, AutoTokenizer

# Apply compatibility patch for lm-eval
import transformers
if not hasattr(transformers, 'Qwen2AudioForConditionalGeneration'):
    class DummyAudioModel:
        pass
    transformers.Qwen2AudioForConditionalGeneration = DummyAudioModel

# Now import lm-eval (should work without error)
from lm_eval import evaluator, simple_evaluate
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance

# Import accelerated_inference modules
from accelerated_inference.kvpress.presses.benchmark_presses import StartRecentKVCache
from accelerated_inference.utils import (
    enable_gpt_neox_pos_shift_attention,
    H2OKVCache,
    LazyH2OKVCache,
)

print("✓ All imports successful with compatibility patch")
```

### Solution 3: Use Alternative Evaluation (If Above Fails)

If neither solution works, you can use a simplified evaluation approach without lm-eval. I can create a standalone evaluation script that doesn't depend on lm-eval.

## Verification

After installing, verify the package is accessible:

```python
# Test import
import accelerated_inference
from accelerated_inference import H2OKVCache, LazyH2OKVCache

print(f"accelerated_inference version: {accelerated_inference.__version__}")
print("✓ Package installed successfully!")
```

## Usage in Notebooks

Once installed with `pip install -e .`, you can import from anywhere:

```python
from accelerated_inference.utils import (
    H2OKVCache,
    LazyH2OKVCache,
    enable_gpt_neox_pos_shift_attention,
)
from accelerated_inference.kvpress.presses.benchmark_presses import (
    StartRecentKVCache,
    SepLLMKVCache,
)
```

## Troubleshooting

**Import errors**: Make sure you're in the correct environment where you ran `pip install -e .`

**lm-eval still fails**: Try Solution 1 (downgrade lm-eval) as it's the most reliable fix

**Module not found**: Run `pip install -e .` from the project root directory (e:/project/accelerated_inference)
