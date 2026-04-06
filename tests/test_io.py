import torch

from easyevo2.io import (
    cleanup_individual_files,
    get_tensor_metadata,
    load_tensor,
    merge_embedding_files,
    save_embeddings,
    save_tensor,
)


def test_save_load_tensor_roundtrip(tmp_path):
    """Save and load a single tensor."""
    tensor = torch.randn(10, 20)
    filepath = tmp_path / "test.safetensors"
    save_tensor(tensor, filepath, tensor_name="my_tensor")

    loaded = load_tensor(filepath, tensor_name="my_tensor", device="cpu")
    assert torch.allclose(tensor, loaded)


def test_save_load_embeddings_roundtrip(tmp_path):
    """Save and load a dictionary of embeddings."""
    embeddings = {"seq1": torch.randn(64), "seq2": torch.randn(64)}
    filepath = tmp_path / "embeddings.safetensors"
    save_embeddings(embeddings, filepath)

    loaded = load_tensor(filepath, device="cpu")
    assert isinstance(loaded, dict)
    assert set(loaded.keys()) == {"seq1", "seq2"}
    assert torch.allclose(embeddings["seq1"], loaded["seq1"])


def test_get_tensor_metadata(tmp_path):
    """Metadata should be saved and retrievable."""
    tensor = torch.randn(5)
    filepath = tmp_path / "meta.safetensors"
    metadata = {"model": "test", "layer": "block.0"}
    save_tensor(tensor, filepath, metadata=metadata)

    retrieved = get_tensor_metadata(filepath)
    assert retrieved["model"] == "test"
    assert retrieved["layer"] == "block.0"


def test_load_tensor_missing_file(tmp_path):
    """Loading from a nonexistent file should raise FileNotFoundError."""
    import pytest

    with pytest.raises(FileNotFoundError):
        load_tensor(tmp_path / "nonexistent.safetensors", device="cpu")


def test_load_tensor_missing_key(tmp_path):
    """Loading a nonexistent tensor name should raise KeyError."""
    import pytest

    tensor = torch.randn(5)
    filepath = tmp_path / "test.safetensors"
    save_tensor(tensor, filepath, tensor_name="exists")

    with pytest.raises(KeyError):
        load_tensor(filepath, tensor_name="does_not_exist", device="cpu")


def test_load_tensor_default_cpu(tmp_path):
    """Default device should be CPU when not specified."""
    tensor = torch.randn(5)
    filepath = tmp_path / "test.safetensors"
    save_tensor(tensor, filepath)

    loaded = load_tensor(filepath)
    assert isinstance(loaded, dict)
    for v in loaded.values():
        assert v.device == torch.device("cpu")


def test_merge_embedding_files(tmp_path):
    """Merging multiple files should combine all embeddings."""
    emb1 = {"seq1": torch.randn(32)}
    emb2 = {"seq2": torch.randn(32)}
    f1 = tmp_path / "part1.safetensors"
    f2 = tmp_path / "part2.safetensors"
    save_embeddings(emb1, f1)
    save_embeddings(emb2, f2)

    merged_path = tmp_path / "merged.safetensors"
    merge_embedding_files([f1, f2], merged_path)

    loaded = load_tensor(merged_path, device="cpu")
    assert "seq1" in loaded
    assert "seq2" in loaded


def test_cleanup_individual_files(tmp_path):
    """Cleanup should delete all listed files."""
    f1 = tmp_path / "a.safetensors"
    f2 = tmp_path / "b.safetensors"
    save_embeddings({"x": torch.randn(4)}, f1)
    save_embeddings({"y": torch.randn(4)}, f2)
    assert f1.exists()
    assert f2.exists()

    cleanup_individual_files([f1, f2])
    assert not f1.exists()
    assert not f2.exists()
