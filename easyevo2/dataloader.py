from collections.abc import Callable
from pathlib import Path

import pyfastx
import torch
from torch.utils.data import DataLoader, Dataset


def get_seq_from_fx(filename: Path | str):
    """
    Read a FASTA or FASTQ file using pyfastx.

    Parameters
    ----------
    filename : Path or str
        Path to the FASTA or FASTQ file.

    Yields
    ------
    tuple
        Name, sequence, and quality score (if FASTQ) for each entry.
    """
    fx = pyfastx.Fastx(filename)
    yield from fx


def get_seq_from_fx_to_dict(filename: Path | str) -> dict[str, str]:
    """
    Read a FASTA or FASTQ file using pyfastx.

    Parameters
    ----------
    filename : Path or str
        Path to the FASTA or FASTQ file.

    Yields
    ------
    tuple
        Name, sequence, and quality score (if FASTQ) for each entry.
    """
    fx = pyfastx.Fastx(filename)
    return dict(fx)


class FxDataset(Dataset):
    """
    PyTorch Dataset for FASTA/FASTQ files.

    Parameters
    ----------
    fx_file : Path or str
        Path to the FASTA or FASTQ file.
    max_length : int | None
        Maximum sequence length. Longer sequences will be truncated.
    transform : Callable | None
        Optional transform to be applied on a sequence.
    preload : bool
        If True, preload all sequences into memory for faster access.
    """

    def __init__(
        self,
        fx_file: Path | str,
        max_length: int | None = None,
        transform: Callable | None = None,
        *,
        preload: bool = False,
    ):
        self.fx_file = str(fx_file)
        self.fx = pyfastx.Fastx(self.fx_file)
        self.names = []
        self.indices = []
        self.sequences = {} if preload else None

        # Pre-process file to get sequence names and indices
        for i, (name, seq, _) in enumerate(self.fx):
            self.names.append(name)
            self.indices.append(i)

            # Preload sequences if requested
            if preload:
                if max_length and len(seq) > max_length:
                    seq = seq[:max_length]
                self.sequences[name] = seq

        self.max_length = max_length
        self.transform = transform
        self.preload = preload

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.names)

    def __getitem__(self, idx: int) -> dict[str, str | torch.Tensor | int]:
        """
        Get a sequence by index.

        Parameters
        ----------
        idx : int
            Index of the sequence to get.

        Returns
        -------
        dict
            Dictionary containing the sequence name, sequence, and original index.
        """
        # Get the name for the given index
        name = self.names[idx]

        # Get sequence - either from preloaded cache or by reading from file
        if self.preload:
            seq = self.sequences[name]
        else:
            # Get the actual sequence using pyfastx
            # Use more optimized approach by accessing Fastx with index
            for i, (_, seq, _) in enumerate(pyfastx.Fastx(self.fx_file)):
                if i == self.indices[idx]:
                    break

            # Truncate sequence if necessary
            if self.max_length and len(seq) > self.max_length:
                seq = seq[: self.max_length]

        # Apply transformations if provided
        if self.transform:
            seq = self.transform(seq)

        return {"name": name, "sequence": seq, "index": self.indices[idx]}


class FxDataLoader:
    """
    PyTorch Data loader for FASTA/FASTQ files.

    This class provides a convenient way to load FASTA/FASTQ files
    for use with PyTorch models to generate embeddings.

    Parameters
    ----------
    fx_file : Path or str
        Path to the FASTA or FASTQ file.
    batch_size : int
        How many samples per batch to load.
    shuffle : bool
        Whether to shuffle the data.
    max_length : int | None
        Maximum sequence length. Longer sequences will be truncated.
    num_workers : int
        How many subprocesses to use for data loading.
    transform : Callable | None
        Optional transform to be applied on a sequence.
    collate_fn : Callable | None
        Merges a list of samples to form a mini-batch.
    pin_memory : bool
        If True, the data loader will copy Tensors into CUDA pinned memory.
    preload : bool
        If True, preload all sequences into memory for faster access.
    prefetch_factor : int
        Number of batches to prefetch if num_workers > 0.
    use_gpu : bool
        If True and GPU is available, utilize CUDA for data processing.
    """

    def __init__(
        self,
        fx_file: Path | str,
        batch_size: int = 32,
        max_length: int | None = None,
        num_workers: int = 0,
        transform: Callable | None = None,
        collate_fn: Callable | None = None,
        *,
        shuffle: bool = False,
        pin_memory: bool = False,
        preload: bool = False,
        prefetch_factor: int = 2,
        use_gpu: bool = False,
    ):
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )

        self.dataset = FxDataset(
            fx_file=fx_file, max_length=max_length, transform=transform, preload=preload
        )

        # Use pin_memory only when using CUDA
        should_pin = pin_memory and self.device.type == "cuda"

        # Set persistent_workers to True if using num_workers to avoid startup overhead
        persistent_workers = num_workers > 0

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn or self._default_collate_fn,
            pin_memory=should_pin,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )

        self.use_gpu = use_gpu and torch.cuda.is_available()

    def _default_collate_fn(
        self, batch: list[dict]
    ) -> dict[str, list[str] | torch.Tensor]:
        """
        Default collate function that handles sequence data.

        Parameters
        ----------
        batch : list of dict
            List of dictionaries, each containing a sequence sample.

        Returns
        -------
        dict
            Dictionary with batched data.
        """
        names = [item["name"] for item in batch]
        sequences = [item["sequence"] for item in batch]
        indices = [item["index"] for item in batch]

        # If sequences are already tensors, stack them and potentially move to GPU
        if isinstance(sequences[0], torch.Tensor):
            sequences_tensor = torch.stack(sequences)
            if self.use_gpu:
                sequences_tensor = sequences_tensor.to(self.device)
            indices_tensor = torch.tensor(
                indices, device=self.device if self.use_gpu else None
            )
            return {
                "names": names,
                "sequences": sequences_tensor,
                "indices": indices_tensor,
            }

        # If sequences are not tensors, just return them as is
        return {"names": names, "sequences": sequences, "indices": indices}

    def __iter__(self):
        """Return an iterator over the dataloader."""
        return iter(self.dataloader)

    def __len__(self) -> int:
        """Return the number of batches in the dataloader."""
        return len(self.dataloader)


# Utility function for GPU memory management
def get_gpu_memory_info():
    """
    Get GPU memory usage information.

    Returns
    -------
    dict
        Dictionary containing GPU memory information.
    """
    if not torch.cuda.is_available():
        return {"available": False}

    # Get GPU count
    gpu_count = torch.cuda.device_count()

    info = {"available": True, "gpu_count": gpu_count, "devices": {}}

    # Get memory info for each GPU
    for i in range(gpu_count):
        total_memory = torch.cuda.get_device_properties(i).total_memory
        reserved_memory = torch.cuda.memory_reserved(i)
        allocated_memory = torch.cuda.memory_allocated(i)
        free_memory = total_memory - reserved_memory

        info["devices"][i] = {
            "total_memory_bytes": total_memory,
            "reserved_memory_bytes": reserved_memory,
            "allocated_memory_bytes": allocated_memory,
            "free_memory_bytes": free_memory,
            "total_memory_gb": total_memory / 1e9,
            "free_memory_gb": free_memory / 1e9,
        }

    return info
