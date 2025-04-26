from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


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

    # Create a dict mapping tensor names to tensors
    tensors_dict = {tensor_name: tensor}

    # Save the tensor
    save_file(tensors_dict, filepath, metadata)


def load_tensor(
    filepath: str | Path,
    tensor_name: str | None = None,
    device: str | int | None = None,
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

    # If tensor_name is provided, load just that tensor
    if tensor_name is not None:
        with safe_open(filepath, framework="pt", device=device) as f:
            if tensor_name not in f:
                msg = f"Tensor '{tensor_name}' not found in {filepath}"
                raise KeyError(msg)
            return f.get_tensor(tensor_name)

    # Otherwise load all tensors
    device = "cpu" if device is None else device
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

    # Save the embeddings
    save_file(embeddings, filepath, metadata)
