from collections.abc import Generator, Iterable

import torch


def check_cuda(device: str) -> None:
    """Check if the specified GPU is available."""
    if device.startswith("cuda") and torch.cuda.is_available():
        # Check if the specified GPU is available
        gpu_index = int(device.split(":")[1])
        if gpu_index >= torch.cuda.device_count():
            msg = f"GPU index {gpu_index} is out of range. Available GPUs: {torch.cuda.device_count()}"
            raise ValueError(msg)


def sliding_window(
    sequences: Iterable[tuple[str, str]], window_size: int, step_size: int
) -> Generator[tuple[str, str]]:
    """
    Slide a window of size `window_size` over the sequences with a step size of `step_size`.

    Return a generator of tuples of the form (name, sequence).
    """
    for name, seq in sequences:
        for i in range(0, len(seq) - window_size + 1, step_size):
            yield f"{name}_{i}_{i + window_size}", seq[i : i + window_size]
