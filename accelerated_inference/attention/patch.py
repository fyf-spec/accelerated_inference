import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

def attention_patch(func):
    """
    Decorator to update the keys before the attention computation.
    For KnormPress (pruning), this might not be strictly necessary if we modify the cache directly.
    But for some methods that mask keys instead of removing them, this is needed.
    """
    def wrapper(module, query, key, value, attention_mask, dropout, **kwargs):
        # Placeholder for future logic if we implement masking-based presses
        return func(module, query, key, value, attention_mask, dropout, **kwargs)
    return wrapper

def patch_attention_functions():
    """
    Apply attention patching to all transformer attention functions.
    """
    for name, func in ALL_ATTENTION_FUNCTIONS.items():
        ALL_ATTENTION_FUNCTIONS[name] = attention_patch(func)
