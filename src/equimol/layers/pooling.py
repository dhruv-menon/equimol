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


def global_mean_pool(h: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean-pool node states into graph states.

    Args:
        h: Node states with shape [N, hidden_dim].
        batch: Optional graph ids with shape [N].

    Returns:
        Graph states with shape [B, H].

    Notes:
        Mean pooling is permutation invariant. It is often a better default for
        graph-level targets that should not scale directly with the number of
        atoms or residues.
    """

    if batch is None:
        return h.mean(dim = 0, keepdim = True) # [1, H]

    num_graphs = int(batch.max().item()) + 1
    pooled_sum = segment_sum(h, batch, num_graphs) # [B, H]
    ones = torch.ones((h.size(0), 1), dtype = h.dtype, device = h.device) # [N, 1]
    counts = segment_sum(ones, batch, num_graphs) # [B, 1]
    pooled_mean = pooled_sum / counts
    return pooled_mean
