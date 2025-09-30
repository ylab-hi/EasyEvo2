import gc
import logging
from collections.abc import Generator, Iterable

import torch
from rich.logging import RichHandler

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger("easyevo2")


def check_cuda(device: str) -> None:
    """Check if the specified GPU is available."""
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            msg = "CUDA is not available on this system"
            raise ValueError(msg)

        # Handle both "cuda" and "cuda:X" formats
        if ":" in device:
            try:
                gpu_index = int(device.split(":")[1])
                if gpu_index >= torch.cuda.device_count():
                    msg = f"GPU index {gpu_index} is out of range. Available GPUs: {torch.cuda.device_count()}"
                    raise ValueError(msg)
            except (ValueError, IndexError):
                msg = f"Invalid CUDA device format: {device}. Expected format: 'cuda' or 'cuda:X'"
                raise ValueError(msg) from None
        else:
            # Just "cuda" - check if any GPU is available
            if torch.cuda.device_count() == 0:
                msg = "No CUDA devices available"
                raise ValueError(msg)


def clear_gpu_memory(device: str) -> None:
    """Clear GPU memory and run garbage collection."""
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()


def validate_sequence(
    seq: str, min_length: int = 1, max_length: int | None = None
) -> bool:
    """
    Validate a sequence for processing.

    Args:
        seq: The sequence to validate
        min_length: Minimum allowed sequence length
        max_length: Maximum allowed sequence length (None for no limit)

    Returns
    -------
        True if sequence is valid, False otherwise
    """
    if not seq or len(seq) < min_length:
        return False

    if max_length and len(seq) > max_length:
        return False

    # Check for valid characters (basic DNA/RNA/protein characters)
    valid_chars = set("ACGTUacgtuNnXx-")
    return all(c in valid_chars for c in seq)


def get_memory_usage(device: str) -> dict[str, float]:
    """
    Get current GPU memory usage for monitoring.

    Args:
        device: Device string ('cpu' or 'cuda:X')

    Returns
    -------
        Dictionary with memory usage information
    """
    memory_info = {}

    if device.startswith("cuda") and torch.cuda.is_available():
        memory_info["gpu_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_info["gpu_reserved"] = torch.cuda.memory_reserved() / 1024**3  # GB
        memory_info["gpu_total"] = (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        )  # GB

    return memory_info


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

    for seq_data in sequences:
        name = seq_data[0]
        seq = seq_data[1]

        # Validate sequence
        if not seq:
            continue

        # return sequence itself first
        yield (name, seq)

        if len(seq) < window_size:
            continue  # Skip sequences shorter than window_size

        for i in range(0, len(seq) - window_size + 1, step_size):
            if use_sequence_without_windows:
                # Only yield if the resulting sequence would be non-empty
                if i > 0 or i + window_size < len(seq):
                    # Use string slicing instead of concatenation for better performance
                    result_seq = seq[:i] + seq[i + window_size :]
                    if result_seq:  # Only yield non-empty sequences
                        yield (
                            f"{name}_without_{i}_{i + window_size}",
                            result_seq,
                        )
            else:
                yield f"{name}_{i}_{i + window_size}", seq[i : i + window_size]
