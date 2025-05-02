from pathlib import Path

import torch

from easyevo2.dataloader import get_seq_from_fx
from easyevo2.io import load_tensor, save_embeddings


class TopyTokenizer:
    """A toy tokenizer that generates random token IDs for testing purposes."""

    def __init__(self):
        self.vocab_size = 10000

    def tokenize(self, sequence) -> list[int]:
        """Tokenize a sequence into random token IDs."""
        return [ord(seq) for seq in sequence]


class ToyModel:
    """A toy model that generates random embeddings for testing purposes."""

    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        self.tokenizer = TopyTokenizer()

    def generate_embedding(self):
        """Generate a random embedding."""
        return torch.randn(self.embedding_dim)

    def __call__(self, *args, **kwds):
        embeddings = {}
        layer_names = kwds.get("layer_names", [])
        for layer_name in layer_names:
            embeddings[layer_name] = self.generate_embedding()

        output = torch.tensor([])
        return output, embeddings


def test_read_fx():
    test_file = "tests/data/test.fa"
    sequences_generator = get_seq_from_fx(test_file)
    sequences = list(sequences_generator)
    assert len(sequences) == 14


def test_model_embed():
    """Test the model loading and embedding generation."""
    # Create a toy model
    model = ToyModel(embedding_dim=128)
    model_type = "toy_model"
    layer_name = ["blocks.28.mlp.l3", "blocks.28.mlp.l4"]
    output = "tests/data/test_embeddings"
    test_file = "tests/data/test.fa"
    sequences = get_seq_from_fx(test_file)
    seq_count = 0
    embeddings_with_name = {}

    # Process sequences in batches
    for name, seq in sequences:
        seq_count += 1
        # Tokenize and process the sequence
        input_ids = torch.tensor(
            model.tokenizer.tokenize(seq),
            dtype=torch.int,
        ).unsqueeze(0)

        with torch.inference_mode():
            # Get embeddings
            outputs, embeddings = model(
                input_ids, return_embeddings=True, layer_names=layer_name
            )

            # Store the embeddings
            # Move embeddings to CPU to free GPU memory
            cpu_embeddings = {
                layer: tensor.detach().cpu() for layer, tensor in embeddings.items()
            }

            # Store the embeddings
            embeddings_with_name[name] = cpu_embeddings

    layer_outputs = []

    # Save the embeddings to the output file
    for layer in layer_name:
        metadata = {
            "output": str(output),
            "layer": layer,
        }

        if output is None:
            layer_output = Path(test_file).with_suffix(
                f".{model_type}.{layer}.safetensors"
            )
        else:
            layer_output = Path(output).with_suffix(
                f".{model_type}.{layer}.safetensors"
            )

        layer_outputs.append(layer_output)

        layer_embeddings = {
            name: embeddings[layer].cpu()
            for name, embeddings in embeddings_with_name.items()
        }

        save_embeddings(
            layer_embeddings,
            layer_output,
            metadata=metadata,
        )

    # Check if the output files are created
    for layer_output in layer_outputs:
        assert layer_output.exists()
        assert layer_output.stat().st_size > 0

        layer_embeddings = load_tensor(layer_output)
        assert isinstance(layer_embeddings, dict)
        assert all(
            isinstance(embedding, torch.Tensor)
            for embedding in layer_embeddings.values()
        )
        assert len(layer_embeddings) == seq_count

        Path(layer_output).unlink()
