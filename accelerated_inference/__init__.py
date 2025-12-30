"""
Accelerated Inference: Efficient KV Cache Strategies for LLM Inference

This package provides implementations of various KV cache compression strategies:
- H2O: Heavy-Hitter Oracle
- Lazy H2O: Periodic H2O updates
- StreamingLLM: Sink + Recent window
- SepLLM: Separator-aware eviction
- UnifiedKVCache: Combined strategies
"""

__version__ = "0.1.0"

# Import main utilities
from .utils import (
    enable_gpt_neox_pos_shift_attention,
    H2OKVCache,
    LazyH2OKVCache,
)

# Import KV cache presses
try:
    from .kvpress.presses.benchmark_presses import StartRecentKVCache, SepLLMKVCache
    from .kvpress.presses.unified_press import UnifiedKVCache
except ImportError:
    # Fallback if kvpress module structure is different
    pass

__all__ = [
    "__version__",
    "enable_gpt_neox_pos_shift_attention",
    "H2OKVCache",
    "LazyH2OKVCache",
    "StartRecentKVCache",
    "SepLLMKVCache",
    "UnifiedKVCache",
]
