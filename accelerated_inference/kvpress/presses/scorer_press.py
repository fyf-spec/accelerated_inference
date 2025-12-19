import logging
from dataclasses import dataclass
import torch
from torch import nn
from kvpress.base_press import BasePress

logger = logging.getLogger(__name__)

@dataclass
class ScorerPress(BasePress):
    """
    Base class for score-based KV cache compression methods.
    """
    compression_ratio: float = 0.0

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        raise NotImplementedError

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

        # Compute scores
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        # Get indices of KV pairs with the lowest scores
        k_len = keys.shape[2]
        n_kept = int(k_len * (1 - self.compression_ratio))
        
        # scores shape: (bsz, num_heads, seq_len)
        # We need to select top k indices along the sequence dimension
        indices = scores.topk(n_kept, dim=-1).indices
        
        # Expand indices to match head_dim
        # indices: (bsz, num_heads, n_kept)
        # keys: (bsz, num_heads, seq_len, head_dim)
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])

        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values
