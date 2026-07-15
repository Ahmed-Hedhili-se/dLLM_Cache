import torch


def _validate_indices(indices: torch.Tensor, name: str = "indices") -> None:
    if indices.dtype != torch.long:
        raise TypeError(f"{name} must be torch.long, got {indices.dtype}")
    if indices.dim() != 2:
        raise ValueError(f"{name} must have shape [B, S], got {tuple(indices.shape)}")


def gather_tokens(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    _validate_indices(indices)
    B, T = tensor.shape[:2]
    S = indices.shape[1]
    expanded_indices = indices.view(B, S, *([1] * (tensor.dim() - 2)))
    expanded_indices = expanded_indices.expand(-1, -1, *tensor.shape[2:])

    return torch.gather(tensor, dim=1, index=expanded_indices)


def scatter_tokens(full_tensor: torch.Tensor, partial_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    _validate_indices(indices)
    B, T = full_tensor.shape[:2]
    S = indices.shape[1]

    expanded_indices = indices.view(B, S, *([1] * (full_tensor.dim() - 2)))
    expanded_indices = expanded_indices.expand(-1, -1, *full_tensor.shape[2:])

    updated_tensor = full_tensor.clone()
    updated_tensor.scatter_(dim=1, index=expanded_indices, src=partial_tensor)

    return updated_tensor