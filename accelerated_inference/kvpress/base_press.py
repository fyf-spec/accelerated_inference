import logging
from dataclasses import dataclass
import torch
from torch import nn
from transformers import PreTrainedModel, QuantizedCache
from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)

@dataclass
class BasePress:
    """
    Base class for all KV cache compression methods.
    """

    def post_init_from_model(self, model: PreTrainedModel):
        pass

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("compress method must be implemented in subclass")

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        # GPTNeoXAttention forward signature:
        # hidden_states, attention_mask, position_ids, layer_past, use_cache, output_attentions
        
        if "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
        elif len(input) > 0:
            hidden_states = input[0]
        else:
            return output

        # Check for cache
        cache = None
        if "layer_past" in kwargs:
            cache = kwargs["layer_past"]
        elif "past_key_values" in kwargs:
            cache = kwargs["past_key_values"]
        elif len(input) > 3:
            cache = input[3]
        
        if cache is None:
            return output
        
        # Let's try to identify the layer index.
        if hasattr(module, "layer_idx"):
            layer_idx = module.layer_idx
        else:
            # Fallback: try to find layer index from cache if possible, or skip
            return output

        q_len = hidden_states.shape[1]
        
        # Simple prefill check: if q_len > 1, we assume prefill.
        if q_len == 1:
            return output

        # Extract keys and values
        # Note: This logic depends heavily on the cache structure.
        
        # Debugging
        # print(f"DEBUG: Cache type: {type(cache)}")
        # print(f"DEBUG: Cache dir: {dir(cache)}")

        if hasattr(cache, "layers"):
             # KVPress reference style
             # Assuming cache.layers is a list of objects with keys/values attributes
             try:
                 cache_layer = cache.layers[layer_idx]
                 keys = cache_layer.keys
                 values = cache_layer.values
             except Exception as e:
                 print(f"DEBUG: Error accessing cache.layers: {e}")
                 raise e
        elif hasattr(cache, "update"): # Duck typing for Cache object
             if hasattr(cache, "key_cache"):
                 keys = cache.key_cache[layer_idx]
                 values = cache.value_cache[layer_idx]
             else:
                 # Fallback for DynamicCache in some versions or other Cache types
                 try:
                     keys, values = cache[layer_idx]
                 except Exception as e:
                     print(f"Error accessing cache via index: {e}")
                     raise e
        else:
             # Tuple cache (key, value)
             keys, values = cache
        
        # Compress
        keys, values = self.compress(module, hidden_states, keys, values, output[1] if len(output) > 1 else None, kwargs)
        
        # Update cache
        if hasattr(cache, "layers"):
             cache_layer = cache.layers[layer_idx]
             cache_layer.keys = keys
             cache_layer.values = values
        elif hasattr(cache, "update"):
            if hasattr(cache, "key_cache"):
                cache.key_cache[layer_idx] = keys
                cache.value_cache[layer_idx] = values
            else:
                # Try to update via index if it supports item assignment
                try:
                    if isinstance(cache, list):
                        cache[layer_idx] = (keys, values)
                    else:
                        if hasattr(cache, "_key_cache"):
                             cache._key_cache[layer_idx] = keys
                             cache._value_cache[layer_idx] = values
                        else:
                             cache[layer_idx] = (keys, values)
                except Exception as e:
                    print(f"DEBUG: Failed to update cache: {e}")
        else:
            # If it was a tuple, we need to update the output
            # Output of attention is usually (attn_output, attn_weights, layer_past)
            # We need to reconstruct layer_past
            new_layer_past = (keys, values)
            output_list = list(output)
            # Find where layer_past is in output. Usually index 2 if weights are returned, or index 1.
            # GPTNeoX: (attn_output, present) or (attn_output, attn_weights, present)
            if len(output_list) == 2:
                output_list[1] = new_layer_past
            elif len(output_list) == 3:
                output_list[2] = new_layer_past
            output = tuple(output_list)

        return output

    def __call__(self, model: PreTrainedModel):
        self.post_init_from_model(model)
        hooks = []
        
        # Identify layers for GPTNeoX
        if hasattr(model, "gpt_neox"):
            layers = model.gpt_neox.layers
        else:
            layers = model.model.layers if hasattr(model, "model") else []

        for layer in layers:
            if hasattr(layer, "attention"):
                hooks.append(layer.attention.register_forward_hook(self.forward_hook, with_kwargs=True))
        
        class Context:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_value, traceback):
                for hook in hooks:
                    hook.remove()
        
        return Context()
