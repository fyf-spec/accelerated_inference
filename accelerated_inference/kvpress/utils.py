import torch
from transformers import Cache, QuantizedCache

def extract_keys_and_values(cache: Cache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts the keys and values from a given cache layer,
    handling both quantized and unquantized caches.
    """
    if isinstance(cache, QuantizedCache):
        # Simplified dequantization for now, assuming standard QuantizedCache usage if applicable
        # But Pythia/NeoX might not use QuantizedCache by default.
        # We'll stick to the reference implementation logic.
        keys = cache.layers[layer_idx]._dequantize(cache.layers[layer_idx]._quantized_keys)
        values = cache.layers[layer_idx]._dequantize(cache.layers[layer_idx]._quantized_values)
    elif hasattr(cache, "key_cache"): # Legacy tuple cache or similar? No, Cache object has .key_cache in some versions?
        # Transformers Cache object usually has .keys and .values lists or similar structure depending on version
        # But DynamicCache stores in a list of tensors.
        # Let's assume DynamicCache which has key_cache and value_cache lists
        if hasattr(cache, "key_cache"):
             keys = cache.key_cache[layer_idx]
             values = cache.value_cache[layer_idx]
        else:
             # Fallback for other Cache implementations
             # The reference implementation uses cache.layers[layer_idx].keys which suggests a specific Cache structure
             # Let's try to be robust.
             try:
                keys = cache.layers[layer_idx].keys
                values = cache.layers[layer_idx].values
             except AttributeError:
                 # DynamicCache in recent transformers
                 keys = cache[layer_idx][0]
                 values = cache[layer_idx][1]
    else:
        # Default to DynamicCache behavior where cache[layer_idx] returns (key, value)
        # Or if it is a list of tuples (legacy)
        keys = cache[layer_idx][0]
        values = cache[layer_idx][1]
        
    return keys, values
