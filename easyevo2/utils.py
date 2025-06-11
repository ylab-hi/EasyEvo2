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
    sequences: Iterable[tuple[str, str]],
    window_size: int,
    step_size: int,
    *,
    use_sequence_without_windows: bool = False,
) -> Generator[tuple[str, str]]:
    """
    Slide a window of size `window_size` over the sequences with a step size of `step_size`.

    Args:
        sequences: Iterable of (name, sequence) tuples
        window_size: Size of the sliding window
        step_size: Number of positions to move the window
        use_sequence_without_windows: If True, yield sequences with windows removed instead of windows

    Returns
    -------
        Generator of tuples of the form (name, sequence)

    Raises
    ------
        ValueError: If window_size or step_size is invalid
    """
    if window_size < 1:
        msg = "window_size must be at least 1"
        raise ValueError(msg)
    if step_size < 1:
        msg = "step_size must be at least 1"
        raise ValueError(msg)

    for name, seq in sequences:
        if len(seq) < window_size:
            continue  # Skip sequences shorter than window_size

        for i in range(0, len(seq) - window_size + 1, step_size):
            if use_sequence_without_windows:
                # Only yield if the resulting sequence would be non-empty
                if i > 0 or i + window_size < len(seq):
                    yield (
                        f"{name}_without_{i}_{i + window_size}",
                        seq[:i] + seq[i + window_size :],
                    )
            else:
                yield f"{name}_{i}_{i + window_size}", seq[i : i + window_size]
