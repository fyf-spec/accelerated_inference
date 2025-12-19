from dataclasses import dataclass
import torch
from torch import nn


# =============================================================================
# StreamingLLM KV Cache (Standalone, works with past_key_values directly)
# =============================================================================

def slice2d(x, start, end):
    """Slice tensor on dimension 2 (typical KV cache sequence dimension)."""
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    """Slice tensor on dimension 3."""
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    """Slice tensor on dimension 1."""
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class StartRecentKVCache:
    """
    StreamingLLM KV Cache that keeps initial tokens ("sinks") and recent tokens.
    
    This is a standalone implementation that works directly with past_key_values
    from HuggingFace models. Use with enable_gpt_neox_pos_shift_attention for
    best results with long sequences.
    
    Args:
        start_size: Number of initial tokens to keep (attention sinks)
        recent_size: Number of recent tokens to keep (sliding window)
        k_seq_dim: Sequence dimension in key tensors (2 for GPT-NeoX/LLaMA)
        v_seq_dim: Sequence dimension in value tensors (2 for GPT-NeoX/LLaMA)
    
    Example:
        >>> kv_cache = StartRecentKVCache(start_size=4, recent_size=252)
        >>> # During generation loop:
        >>> outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
        >>> past_key_values = kv_cache(outputs.past_key_values)
    """
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

    def __call__(self, past_key_values):
        """
        Apply cache eviction to past_key_values.
        
        Keeps start_size initial tokens and recent_size recent tokens,
        evicting tokens in the middle when cache exceeds cache_size.
        """
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_for_space(self, past_key_values, num_coming):
        """
        Evict tokens to make space for num_coming new tokens.
        
        Use this to proactively evict before adding new tokens, rather than
        after (which is what __call__ does).
        """
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_range(self, past_key_values, start, end):
        """
        Evict a specific range of tokens from the cache.
        
        Removes tokens from index start to end (exclusive).
        """
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]


# =============================================================================
# KVPress-based Press Implementations
# =============================================================================

# Import BasePress here (after standalone StartRecentKVCache) to avoid circular imports
try:
    from accelerated_inference.kvpress.base_press import BasePress
    HAS_BASE_PRESS = True
except ImportError:
    # Fallback: create a dummy BasePress if kvpress is not available
    HAS_BASE_PRESS = False
    @dataclass
    class BasePress:
        """Dummy BasePress for when kvpress is not available."""
        def compress(self, module, hidden_states, keys, values, attentions, kwargs):
            return keys, values

@dataclass
class StreamLLMPress(BasePress):
    """
    StreamingLLM: Keep initial tokens (sinks) and recent window.
    """
    compression_ratio: float = 0.0
    num_sinks: int = 4

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        seq_len = keys.shape[2]
        n_kept = int(seq_len * (1 - self.compression_ratio))
        
        if n_kept >= seq_len:
            return keys, values
            
        if n_kept <= self.num_sinks:
             # Edge case: kept is smaller than sinks, just keep recent 
             return keys[:, :, -n_kept:], values[:, :, -n_kept:]
        
        # Keep sinks
        sinks_k = keys[:, :, :self.num_sinks]
        sinks_v = values[:, :, :self.num_sinks]
        
        # Keep recent
        window_size = n_kept - self.num_sinks
        recent_k = keys[:, :, -window_size:]
        recent_v = values[:, :, -window_size:]
        
        return torch.cat([sinks_k, recent_k], dim=2), torch.cat([sinks_v, recent_v], dim=2)

@dataclass
class SnapKVPress(BasePress):
    """
    SnapKV: Select important KV pairs based on attention scores from a 'window' of observation.
    Simplified implementation for benchmarking.
    """
    compression_ratio: float = 0.0
    window_size: int = 32 # Observation window size
    kernel_size: int = 5 
    
    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.compression_ratio == 0:
            return keys, values
            
        seq_len = keys.shape[2]
        n_kept = int(seq_len * (1 - self.compression_ratio))
        
        # If we don't have attention scores (e.g. prefill), we can't prune effectively using SnapKV logic usually
        # But here 'attentions' might be passed?
        # In base_press, `output[1]` is passed as `attentions`.
        # If None, we cannot compress based on attention.
        
        if attentions is None:
            # Fallback to recent window (StreamingLLM style) if no attention scores
            return keys[:, :, -n_kept:], values[:, :, -n_kept:]
            
        # attentions shape: (bsz, num_heads, q_len, k_len)
        # We perform pruning based on the last few tokens' attention to the past
        
        # Take average attention over the observation window (last `window_size` queries)
        # We need to be careful with shapes.
        # If q_len is small (generation), we use it.
        
        # Sum attention over query dimension
        # attention_score: (bsz, num_heads, k_len)
        attention_score = attentions.sum(dim=-2) 
        
        # Select top-k
        indices = attention_score.topk(n_kept, dim=-1).indices
        indices = indices.sort(dim=-1).values # Sort to keep temporal order if needed/preferred
        
        # Gather
        # indices: (bsz, num_heads, n_kept)
        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
        
        keys = keys.gather(2, expanded_indices).contiguous()
        values = values.gather(2, expanded_indices).contiguous()
        
        return keys, values
