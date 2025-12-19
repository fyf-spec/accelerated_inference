from dataclasses import dataclass
import torch
from torch import nn
from kvpress.presses.scorer_press import ScorerPress

@dataclass
class KnormPress(ScorerPress):
    """
    Key norm-based KV cache compression.
    """

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        # keys shape: (bsz, num_heads, seq_len, head_dim)
        # score: L2 norm of key vectors
        return -keys.norm(dim=-1)
