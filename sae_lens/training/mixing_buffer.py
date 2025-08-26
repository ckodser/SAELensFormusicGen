from collections.abc import Iterator

import torch


@torch.no_grad()
def mixing_buffer(
    buffer_size: int,
    batch_size: int,
    activations_loader: Iterator[torch.Tensor],
) -> Iterator[torch.Tensor]:
    """
    A generator that maintains a mix of old and new activations for better training.
    It stores half of the activations and mixes them with new ones to create batches.

    Args:
        buffer_size: Total size of the buffer (will store buffer_size/2 activations)
        batch_size: Size of batches to return
        activations_loader: Iterator providing new activations

    Yields:
        Batches of activations of shape (batch_size, *activation_dims)
    """

    storage_buffer: torch.Tensor | None = None

    for new_activations in activations_loader:
        serving_batches = new_activations.shape[0] // batch_size
        for i in range(serving_batches):
            yield new_activations[i * batch_size : (i + 1) * batch_size].float()
        new_activations = new_activations[serving_batches * batch_size:]
        storage_buffer = (
            new_activations
            if storage_buffer is None
            else torch.cat([storage_buffer, new_activations], dim=0)
        )
        serving_batches = storage_buffer.shape[0] // batch_size
        for i in range(serving_batches):
            yield storage_buffer[i * batch_size : (i + 1) * batch_size].float()
