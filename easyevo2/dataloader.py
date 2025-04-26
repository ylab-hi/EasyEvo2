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
    """

    def __init__(
        self,
        fx_file: Path | str,
        max_length: int | None = None,
        transform: Callable | None = None,
    ):
        self.fx_file = str(fx_file)
        self.fx = pyfastx.Fastx(self.fx_file)
        self.names = []
        self.indices = []

        # Pre-process file to get sequence names and indices
        for i, (name, _, _) in enumerate(self.fx):
            self.names.append(name)
            self.indices.append(i)

        self.max_length = max_length
        self.transform = transform

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

        # Get the actual sequence using pyfastx
        # Note: pyfastx.Fastx is not directly indexable, so we reopen the file
        # and iterate to the correct position
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
    ):
        self.dataset = FxDataset(
            fx_file=fx_file, max_length=max_length, transform=transform
        )

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn or self._default_collate_fn,
            pin_memory=pin_memory,
        )

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

        # If sequences are already tensors, stack them
        if isinstance(sequences[0], torch.Tensor):
            sequences_tensor = torch.stack(sequences)
            return {"names": names, "sequences": sequences_tensor, "indices": indices}

        return {"names": names, "sequences": sequences, "indices": indices}

    def __iter__(self):
        """Return an iterator over the dataloader."""
        return iter(self.dataloader)

    def __len__(self) -> int:
        """Return the number of batches in the dataloader."""
        return len(self.dataloader)
