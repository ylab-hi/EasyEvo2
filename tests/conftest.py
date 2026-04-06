from pathlib import Path

import pytest
import torch


class TopyTokenizer:
    """A toy tokenizer that generates token IDs for testing purposes."""

    def __init__(self):
        self.vocab_size = 10000

    def tokenize(self, sequence: str) -> list[int]:
        """Tokenize a sequence into character ordinals."""
        return [ord(c) for c in sequence]


class ToyModel:
    """A toy model that generates random embeddings for testing purposes."""

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.tokenizer = TopyTokenizer()

    def generate_embedding(self) -> torch.Tensor:
        """Generate a random embedding."""
        return torch.randn(self.embedding_dim)

    def __call__(self, *args, **kwds):
        embeddings = {}
        layer_names = kwds.get("layer_names", [])
        for layer_name in layer_names:
            embeddings[layer_name] = self.generate_embedding()
        output = torch.tensor([])
        return output, embeddings

    def score_sequences(self, sequences: list[str]) -> list[float]:
        """Return fake scores for sequences."""
        return [0.5 * len(seq) / 100.0 for seq in sequences]


@pytest.fixture
def toy_model():
    """Provide a ToyModel instance."""
    return ToyModel(embedding_dim=128)


@pytest.fixture
def test_fasta() -> Path:
    """Path to the test FASTA file."""
    return Path("tests/data/test.fa")


@pytest.fixture
def test_vcf() -> Path:
    """Path to the test VCF file."""
    return Path("tests/data/test.vcf")


@pytest.fixture
def tmp_output(tmp_path) -> Path:
    """Provide a temporary output directory."""
    return tmp_path
