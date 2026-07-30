from __future__ import annotations
from typing import Optional
import torch
from equimol.utils import segment_sum

# ----------------------------------------
# Pooling layers for invariant graph-level readouts.
#    - Converts node states into graph states without depending on node order
#
# Shapes:
#    h: [N, H] invariant node states.
#    batch: optional [N] graph id per node.
#    output: [B, H] graph states.
#
# Summation is permutation invariant within each graph. If each node state is
# invariant to rotations/translations, the pooled graph state is invariant too.
#
# Complexity: O(N x H)
# ----------------------------------------

def global_add_pool(h: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
    '''Sum node states into graph states.

    Args:
        h: Node states with shape [N, hidden_dim].
        batch: Optional graph ids with shape [N].

    Returns:
        Graph states with shape [B, H].'''

    if batch is None: 
        return h.sum(dim = 0, keepdim = True)
    return segment_sum(h, batch, int(batch.max().item()) + 1)
