
from __future__ import annotations
import torch

# ----------------------------------------
# This utility function comes in handy for aggregating messages  
# during message passing & subsequent node / coordinate updates.
# 
# A worked out example of how it works is as follows:
# Consider, src = [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]
#           src has shape: [5, 2] 
# Now, consider index = [0, 1, 0, 2, 1], with shape: [5]
# Finally, consider num_segments = 4
# ---
# Segment sum first constructs output zeros of shape: [num_segments, src.shape[-1]]
# In our worked out example, this is: [4, 2]
# Next, we add the src row into output[index[row]]
# Therefore, segment 0 gets rows 0 and 2 
#            segment 1 gets rows 1 and 4
#            segment 2 gets row 3
#            segment 3 gets no row
# So, our final output looks like: 
# output = [[4, 40], [7, 70], [4, 40], [0, 0]] 
# ----------------------------------------  

def segment_sum(src: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    '''sum rows of src tensor into num_segments groups.

    Args:
        - src: Values with shape = [E, F]
        - index: Segment id per row with shape = [E]
        - num_segments: Number of output groups

    Returns:
        - Tensor with shape = [num_segments, F]'''

    out = torch.zeros(num_segments, src.size(-1), 
                      device = src.device, dtype = src.dtype) # [num_segments, F]
    
    return out.index_add_(0, index, src) 
