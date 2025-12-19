
import torch
import torch.nn as nn
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from typing import Optional, Callable
import math


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GPTNeoXGQA(nn.Module):
    """
    GPT-NeoX Attention with Grouped Query Attention (GQA).
    Mirrors the interface of GPTNeoXAttention but uses fewer KV heads.
    """
    def __init__(self, config, num_kv_heads=8, layer_idx=None):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_kv_heads
        
        self.rotary_ndims = int(self.head_dim * config.rotary_pct)
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.layer_idx = layer_idx
        
        # Query projection: (hidden_size -> num_q_heads * head_dim)
        # K, V projection: (hidden_size -> num_kv_heads * head_dim)
        # Combined into single linear for efficiency, but split differently than original
        # Original: query_key_value outputs [3 * hidden_size] = [num_heads * 3 * head_dim]
        # GQA: query outputs [num_q_heads * head_dim], key and value each output [num_kv_heads * head_dim]
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=getattr(config, 'attention_bias', True))
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=getattr(config, 'attention_bias', True))
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=getattr(config, 'attention_bias', True))
        self.dense = nn.Linear(self.hidden_size, self.hidden_size, bias=getattr(config, 'attention_bias', True))

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        layer_past: Optional[tuple] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple] = None,
        **kwargs,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V separately
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to [batch, num_heads, seq_len, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        # position_embeddings is (cos, sin) from GPTNeoXRotaryEmbedding
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Cache QKV values using the new Cache API (same as official GPTNeoXAttention)
        if layer_past is not None:
            cache_kwargs = {
                "sin": sin if position_embeddings is not None else None,
                "cos": cos if position_embeddings is not None else None,
                "partial_rotation_size": self.rotary_ndims,
                "cache_position": cache_position,
            }
            key_states, value_states = layer_past.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        # Repeat K, V for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout if self.training else 0.0)
        
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        attn_output = self.dense(attn_output)
        
        # Match GPTNeoXAttention return format: (attn_output, attn_weights)
        return attn_output, attn_weights if output_attentions else None


def convert_model_to_gqa(model, num_kv_heads=8):
    """
    Convert a GPTNeoXForCausalLM model to GQA.
    Averages weights for K and V heads within each group.
    """
    config = model.config
    num_query_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_query_heads
    group_size = num_query_heads // num_kv_heads
    
    print(f"Converting model to GQA: {num_query_heads} Q heads, {num_kv_heads} KV heads (group size {group_size})")
    
    for i, layer in enumerate(model.gpt_neox.layers):
        old_attn = layer.attention
        new_attn = GPTNeoXGQA(config, num_kv_heads=num_kv_heads, layer_idx=i).to(model.device).to(model.dtype)
        
        # Old query_key_value weight: [3 * hidden_size, hidden_size]
        # Structure is interleaved: for each head, [Q, K, V] blocks of size head_dim
        # So shape when viewed: [num_heads, 3, head_dim, hidden_size]
        old_weight = old_attn.query_key_value.weight  # [3 * hidden_size, hidden_size]
        old_bias = old_attn.query_key_value.bias if old_attn.query_key_value.bias is not None else None
        
        # Reshape to separate Q, K, V per head
        # [num_heads, 3, head_dim, hidden_size]
        old_w_reshaped = old_weight.view(num_query_heads, 3, head_dim, hidden_size)
        
        q_w = old_w_reshaped[:, 0, :, :]  # [num_heads, head_dim, hidden_size]
        k_w = old_w_reshaped[:, 1, :, :]
        v_w = old_w_reshaped[:, 2, :, :]
        
        # Q: copy as is, reshape to [num_q_heads * head_dim, hidden_size]
        new_q_w = q_w.reshape(num_query_heads * head_dim, hidden_size)
        
        # K, V: group and average
        # [num_kv_heads, group_size, head_dim, hidden_size] -> mean -> [num_kv_heads, head_dim, hidden_size]
        k_w_grouped = k_w.view(num_kv_heads, group_size, head_dim, hidden_size)
        k_w_mean = k_w_grouped.mean(dim=1)
        new_k_w = k_w_mean.reshape(num_kv_heads * head_dim, hidden_size)
        
        v_w_grouped = v_w.view(num_kv_heads, group_size, head_dim, hidden_size)
        v_w_mean = v_w_grouped.mean(dim=1)
        new_v_w = v_w_mean.reshape(num_kv_heads * head_dim, hidden_size)
        
        # Set new weights
        new_attn.q_proj.weight.data = new_q_w
        new_attn.k_proj.weight.data = new_k_w
        new_attn.v_proj.weight.data = new_v_w
        
        # Handle bias if present
        if old_bias is not None:
            old_b_reshaped = old_bias.view(num_query_heads, 3, head_dim)
            q_b = old_b_reshaped[:, 0, :].reshape(num_query_heads * head_dim)
            k_b = old_b_reshaped[:, 1, :].view(num_kv_heads, group_size, head_dim).mean(dim=1).reshape(num_kv_heads * head_dim)
            v_b = old_b_reshaped[:, 2, :].view(num_kv_heads, group_size, head_dim).mean(dim=1).reshape(num_kv_heads * head_dim)
            
            new_attn.q_proj.bias.data = q_b
            new_attn.k_proj.bias.data = k_b
            new_attn.v_proj.bias.data = v_b
        
        # Copy dense weights
        new_attn.dense.weight.data = old_attn.dense.weight.data
        if old_attn.dense.bias is not None:
            new_attn.dense.bias.data = old_attn.dense.bias.data
        
        # Replace attention module
        layer.attention = new_attn
        
    return model
