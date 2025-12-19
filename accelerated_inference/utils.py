"""
StreamingLLM Utilities for accelerated_inference

This module contains utilities from the streaming-llm project adapted for 
GPT-NeoX (Pythia) architecture.
"""

import torch
import argparse
import types
import os
import os.path as osp
import ssl
import urllib.request
import json
from typing import Optional, Tuple

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.models.gpt_neox.modeling_gpt_neox import (
    apply_rotary_pos_emb,
    rotate_half,
    GPTNeoXAttention,
)


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    """Parse command line arguments for streaming LLM evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="models/llama/llama-7b"
    )
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="wikitext")

    parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")
    parser.add_argument(
        "--split", type=str, default="test", choices=["validation", "test"]
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/debug",
    )

    parser.add_argument("--enable_start_recent_kv_cache", action="store_true")
    parser.add_argument("--start_size", type=int, default=1)
    parser.add_argument("--recent_size", type=int, default=255)
    parser.add_argument("--enable_pos_shift", action="store_true")

    parser.add_argument("--num_eval_tokens", type=int, default=None)

    args = parser.parse_args()
    return args


# =============================================================================
# Model Loading
# =============================================================================

def load(model_name_or_path):
    """Load model and tokenizer from path."""
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


# =============================================================================
# Utility Functions
# =============================================================================

def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f"File {file} exists, use existing file.")
        return path

    print(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


def load_jsonl(file_path):
    """Load JSONL file and return list of dictionaries."""
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    return list_data_dict


# =============================================================================
# GPT-NeoX Position Shift Attention (for StreamingLLM)
# =============================================================================

def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    """Apply rotary position embedding to a single tensor (query or key)."""
    # cos: [1, 1, max_seq_len, head_dim]
    # sin: [1, 1, max_seq_len, head_dim]
    # position_ids: [bs, seq_len]
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def gpt_neox_pos_shift_attention_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    position_ids: torch.LongTensor,
    head_mask: Optional[torch.FloatTensor] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
):
    """
    Modified GPT-NeoX attention forward with position shift for StreamingLLM.
    
    This enables position shift for the KV cache so that the model can
    attend to tokens beyond its original context window.
    """
    has_layer_past = layer_past is not None

    # Compute QKV
    # Attention heads [batch, seq_len, hidden_size]
    #   --> [batch, seq_len, (np * 3 * head_size)]
    qkv = self.query_key_value(hidden_states)

    # [batch, seq_len, (num_heads * 3 * head_size)]
    #   --> [batch, seq_len, num_heads, 3 * head_size]
    new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
    qkv = qkv.view(*new_qkv_shape)

    # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
    query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
    key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
    value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

    # Compute rotary embeddings on rotary_ndims
    query_rot = query[..., : self.rotary_ndims]
    query_pass = query[..., self.rotary_ndims :]

    # Compute token offset for rotary embeddings (when decoding)
    seq_len = key.shape[-2]
    if has_layer_past:
        seq_len += layer_past[0].shape[-2]
    cos, sin = self.rotary_emb(value, seq_len=seq_len)
    query = apply_rotary_pos_emb_single(query_rot, cos, sin, position_ids)
    query = torch.cat((query, query_pass), dim=-1)

    # Cache QKV values
    if has_layer_past:
        past_key = layer_past[0]
        past_value = layer_past[1]
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)

    present = (key, value) if use_cache else None

    key_rot = key[..., : self.rotary_ndims]
    key_pass = key[..., self.rotary_ndims :]
    key_position_ids = torch.arange(seq_len, device=position_ids.device).unsqueeze(0)
    key = apply_rotary_pos_emb_single(key_rot, cos, sin, key_position_ids)
    key = torch.cat((key, key_pass), dim=-1)

    # Compute attention
    attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

    # Reshape outputs
    attn_output = self._merge_heads(
        attn_output, self.num_attention_heads, self.head_size
    )
    attn_output = self.dense(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def enable_gpt_neox_pos_shift_attention(model):
    """
    Enable position shift attention for GPT-NeoX model.
    
    This replaces the attention forward method in all GPTNeoXAttention modules
    with the position-shift-aware version for StreamingLLM.
    """
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_gpt_neox_pos_shift_attention(module)

        if isinstance(module, GPTNeoXAttention):
            module.forward = types.MethodType(
                gpt_neox_pos_shift_attention_forward, module
            )


# =============================================================================
# StreamingLLM KV Cache Import (from benchmark_presses)
# =============================================================================

# Import StartRecentKVCache from benchmark_presses for convenience
from accelerated_inference.kvpress.presses.benchmark_presses import StartRecentKVCache


def enable_streaming_llm(model, start_size, recent_size):
    """
    Enable StreamingLLM for a model.
    
    This function:
    1. Enables position shift attention for the model
    2. Returns a KV cache that keeps start and recent tokens
    
    Args:
        model: HuggingFace model (GPT-NeoX/Pythia, LLaMA, MPT, Falcon)
        start_size: Number of initial tokens to keep ("sink" tokens)
        recent_size: Number of recent tokens to keep
        
    Returns:
        StartRecentKVCache instance for the model
    """
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        # LLaMA position shift not included - add if needed
        raise NotImplementedError("LLaMA position shift not implemented in this module")
    elif "mpt" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "gpt_neox" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        enable_gpt_neox_pos_shift_attention(model)
    elif "falcon" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
        # Falcon position shift not included - add if needed
        raise NotImplementedError("Falcon position shift not implemented in this module")
    else:
        raise ValueError(f"Unsupported model type: {model.config.model_type}")
    
    kv_cache = StartRecentKVCache(
        start_size=start_size,
        recent_size=recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
    return kv_cache
