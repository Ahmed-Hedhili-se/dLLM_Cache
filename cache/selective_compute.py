import torch

def gather_tokens(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Gathers tokens from the full sequence based on indices.
    
    Args:
        tensor: [B, T, ...] - The full feature tensor.
        indices: [B, S] - The indices of tokens to gather (S <= T).
        
    Returns:
        gathered_tensor: [B, S, ...] - The gathered feature tensor.
    """
    B, T = tensor.shape[:2]
    S = indices.shape[1]
    
    # Expand indices to match the trailing dimensions of the tensor
    # e.g., if tensor is [B, T, D], indices must become [B, S, D]
    expanded_indices = indices.view(B, S, *([1] * (tensor.dim() - 2)))
    expanded_indices = expanded_indices.expand(-1, -1, *tensor.shape[2:])
    
    return torch.gather(tensor, dim=1, index=expanded_indices)


def scatter_tokens(full_tensor: torch.Tensor, partial_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Scatters partially updated tokens back into the full tensor.
    
    Args:
        full_tensor: [B, T, ...] - The original, full-length feature tensor (will be cloned or modified).
        partial_tensor: [B, S, ...] - The updated features for the selected tokens.
        indices: [B, S] - The indices corresponding to the partial_tensor.
        
    Returns:
        updated_tensor: [B, T, ...] - The full tensor with scattered updates.
    """
    B, T = full_tensor.shape[:2]
    S = indices.shape[1]
    
    expanded_indices = indices.view(B, S, *([1] * (full_tensor.dim() - 2)))
    expanded_indices = expanded_indices.expand(-1, -1, *full_tensor.shape[2:])
    
    # We clone to avoid in-place modification issues, though it could be optimized depending on use case.
    updated_tensor = full_tensor.clone()
    updated_tensor.scatter_(dim=1, index=expanded_indices, src=partial_tensor)
    
    return updated_tensor