from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from easyevo2.utils import log


def save_tensor(
    tensor: torch.Tensor,
    filepath: str | Path,
    tensor_name: str = "tensor",
    metadata: dict[str, str] | None = None,
) -> None:
    """
    Save a PyTorch tensor using the safetensors format.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to save.
    filepath : str or Path
        Path where the tensor will be saved. If extension is not provided,
        '.safetensors' will be added.
    tensor_name : str, default="tensor"
        Name to identify the tensor within the saved file.
    metadata : dict of str to str, optional
        Metadata to associate with the tensor, such as shape, dtype,
        creation date, or any custom information.

    Examples
    --------
    >>> import torch
    >>> tensor = torch.randn(10, 10)
    >>> save_tensor(tensor, "example_tensor.safetensors", "my_matrix")
    >>> # With metadata
    >>> save_tensor(
    ...     tensor,
    ...     "example_tensor.safetensors",
    ...     "my_matrix",
    ...     {"created_by": "user", "purpose": "test"},
    ... )
    """
    # Convert filepath to Path object
    filepath = Path(filepath)

    # Ensure the file has the correct extension
    if filepath.suffix != ".safetensors":
        filepath = filepath.with_suffix(".safetensors")

    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Move tensor to CPU before saving if it's on GPU
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Create a dict mapping tensor names to tensors
    tensors_dict = {tensor_name: tensor}

    # Save the tensor
    save_file(tensors_dict, filepath, metadata)


def load_tensor(
    filepath: str | Path,
    tensor_name: str | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """
    Load a PyTorch tensor from the safetensors format.

    Parameters
    ----------
    filepath : str or Path
        Path to the tensor file.
    tensor_name : str, optional
        Name of the specific tensor to load. If None, all tensors will be loaded.
    device : str or torch.device, optional
        Device to load the tensor(s) to. If None, tensors will be loaded to the
        same device they were saved from.

    Returns
    -------
    torch.Tensor or dict of str to torch.Tensor
        If tensor_name is provided, returns the requested tensor.
        Otherwise, returns a dictionary of all tensors in the file.

    Examples
    --------
    >>> # Load a specific tensor
    >>> tensor = load_tensor("example_tensor.safetensors", "my_matrix")
    >>> # Load all tensors
    >>> tensors_dict = load_tensor("example_tensor.safetensors")
    >>> # Load to specific device
    >>> tensor = load_tensor("example_tensor.safetensors", "my_matrix", "cuda:0")
    """
    # Convert filepath to Path object
    filepath = Path(filepath)

    # Ensure the file has the correct extension
    if filepath.suffix != ".safetensors":
        filepath = filepath.with_suffix(".safetensors")

    # Check if file exists
    if not filepath.exists():
        msg = f"No tensor file found at {filepath}"
        raise FileNotFoundError(msg)

    # Set the device based on user preference or availability
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # If tensor_name is provided, load just that tensor
    if tensor_name is not None:
        with safe_open(filepath, framework="pt", device=device) as f:
            if tensor_name not in f:
                msg = f"Tensor '{tensor_name}' not found in {filepath}"
                raise KeyError(msg)
            return f.get_tensor(tensor_name)

    # Otherwise load all tensors
    return load_file(filepath, device=device)


def get_tensor_metadata(filepath: str | Path) -> dict[str, str]:
    """
    Get metadata associated with tensors in a safetensors file.

    Parameters
    ----------
    filepath : str or Path
        Path to the tensor file.

    Returns
    -------
    dict
        Dictionary of metadata associated with the tensors.

    Examples
    --------
    >>> metadata = get_tensor_metadata("example_tensor.safetensors")
    >>> print(metadata)
    """
    # Convert filepath to Path object
    filepath = Path(filepath)

    # Ensure the file has the correct extension
    if filepath.suffix != ".safetensors":
        filepath = filepath.with_suffix(".safetensors")

    # Check if file exists
    if not filepath.exists():
        msg = f"No tensor file found at {filepath}"
        raise FileNotFoundError(msg)

    # Get metadata
    with safe_open(filepath, framework="pt") as f:
        return f.metadata()


def save_embeddings(
    embeddings: dict[str, torch.Tensor],
    filepath: str | Path,
    metadata: dict[str, str] | None = None,
) -> None:
    """
    Save a dictionary of embeddings using the safetensors format.

    Parameters
    ----------
    embeddings : dict of str to torch.Tensor
        Dictionary mapping identifiers to embedding tensors.
    filepath : str or Path
        Path where the embeddings will be saved.
    metadata : dict of str to str, optional
        Metadata to associate with the embeddings.

    Examples
    --------
    >>> import torch
    >>> embeddings = {"seq1": torch.randn(768), "seq2": torch.randn(768)}
    >>> save_embeddings(embeddings, "sequence_embeddings.safetensors")
    """
    # Convert filepath to Path object
    filepath = Path(filepath)

    # Ensure the file has the correct extension
    if filepath.suffix != ".safetensors":
        filepath = filepath.with_suffix(".safetensors")

    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Move embeddings to CPU before saving if they're on GPU
    cpu_embeddings = {}
    for key, tensor in embeddings.items():
        if tensor.is_cuda:
            cpu_embeddings[key] = tensor.cpu()
        else:
            cpu_embeddings[key] = tensor

    # Save the embeddings
    save_file(cpu_embeddings, filepath, metadata)


def get_optimal_device(
    memory_required: float = 0, *, prefer_gpu: bool = True
) -> torch.device:
    """
    Get the optimal device for tensor operations based on memory requirements.

    Parameters
    ----------
    memory_required : float
        Estimated memory required in GB. If the GPU has less memory, will use CPU.
    prefer_gpu : bool
        If True, prefer using GPU if available regardless of memory requirements.

    Returns
    -------
    torch.device
        The optimal device to use (CUDA or CPU).
    """
    # If CUDA is not available, return CPU
    if not torch.cuda.is_available():
        return torch.device("cpu")

    # If no memory requirement or preference is GPU, use CUDA
    if memory_required <= 0 or not prefer_gpu:
        return torch.device("cuda")

    # Check GPU memory availability
    free_memory = []
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory
        reserved_memory = torch.cuda.memory_reserved(i)
        free_memory.append((total_memory - reserved_memory) / 1e9)  # Convert to GB

    # Find GPU with most free memory
    if free_memory:
        max_free_memory = max(free_memory)
        best_gpu_index = free_memory.index(max_free_memory)

        # If enough memory is available, use that GPU
        if max_free_memory >= memory_required:
            return torch.device(f"cuda:{best_gpu_index}")

    # Default to CPU if no suitable GPU found
    return torch.device("cpu")


def save_single_embedding(
    name: str,
    embedding: torch.Tensor,
    filepath: str | Path,
    metadata: dict[str, str] | None = None,
) -> None:
    """
    Save a single embedding for a specific sequence.

    Parameters
    ----------
    name : str
        Name/identifier of the sequence.
    embedding : torch.Tensor
        The embedding tensor to save.
    filepath : str or Path
        Path where the embedding will be saved.
    metadata : dict of str to str, optional
        Metadata to associate with the embedding.
    """
    # Convert filepath to Path object
    filepath = Path(filepath)

    # Ensure the file has the correct extension
    if filepath.suffix != ".safetensors":
        filepath = filepath.with_suffix(".safetensors")

    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Move embedding to CPU before saving if it's on GPU
    if embedding.is_cuda:
        embedding = embedding.cpu()

    # Create a dict with the single embedding
    embeddings_dict = {name: embedding}

    # Save the embedding
    save_file(embeddings_dict, filepath, metadata)


def merge_embedding_files(
    filepaths: list[str | Path],
    output_filepath: str | Path,
    metadata: dict[str, str] | None = None,
) -> None:
    """
    Merge multiple embedding files into a single file.

    Parameters
    ----------
    filepaths : list of str or Path
        List of paths to embedding files to merge.
    output_filepath : str or Path
        Path where the merged embeddings will be saved.
    metadata : dict of str to str, optional
        Metadata to associate with the merged embeddings.
    """
    # Convert output_filepath to Path object
    output_filepath = Path(output_filepath)

    # Ensure the file has the correct extension
    if output_filepath.suffix != ".safetensors":
        output_filepath = output_filepath.with_suffix(".safetensors")

    # Ensure parent directory exists
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    # Load and merge all embeddings
    merged_embeddings = {}

    for filepath in filepaths:
        filepath = Path(filepath)
        if filepath.exists():
            try:
                embeddings = load_file(filepath)
                merged_embeddings.update(embeddings)
            except Exception as e:
                log.warning(f"Failed to load embeddings from {filepath}: {e}")
                continue

    # Save merged embeddings
    if merged_embeddings:
        save_file(merged_embeddings, output_filepath, metadata)
        log.info(f"Merged embeddings saved to {output_filepath}")
    else:
        log.warning("No embeddings to merge")


def cleanup_individual_files(filepaths: list[str | Path]) -> None:
    """
    Clean up individual embedding files after successful merge.

    Parameters
    ----------
    filepaths : list of str or Path
        List of paths to files to delete.
    """
    for filepath in filepaths:
        filepath = Path(filepath)
        if filepath.exists():
            try:
                filepath.unlink()
            except Exception as e:
                log.warning(f"Failed to delete {filepath}: {e}")
