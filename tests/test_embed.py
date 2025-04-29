from pathlib import Path

import torch

from easyevo2.dataloader import get_seq_from_fx_to_dict
from easyevo2.io import save_embeddings


class ToyModel:
    """A toy model that generates random embeddings for testing purposes."""

    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

    def generate_embedding(self):
        """Generate a random embedding."""
        return torch.randn(self.embedding_dim)

    def __call__(self, layer_names, *args, **kwds):
        embeddings = {}
        for layer_name in layer_names:
            embeddings[layer_name] = self.generate_embedding()

        output = torch.tensor([])
        return output, embeddings


def test_model_embed():
    """Test the model loading and embedding generation."""
    # Create a toy model
    model = ToyModel(embedding_dim=128)
    model_type = "toy_model"
    layer_name = ["blocks.28.mlp.l3"]
    output = "tests/data/test_embeddings"
    test_file = "tests/data/test.fasta"
    sequences = get_seq_from_fx_to_dict()
    embeddings_with_name = {}

    # Process sequences in batches
    for name, seq in sequences.items():
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
            embeddings_with_name[name] = embeddings

    # Save the embeddings to the output file
    for layer in layer_name:
        metadata = {
            "output": str(output),
        }

        if output is None:
            output = Path(test_file).with_suffix(f".{model_type}.{layer}.safetensors")
        else:
            output = Path(output).with_suffix(f".{model_type}.{layer}.safetensors")

        layer_embeddings = {
            name: embeddings[layer].cpu()
            for name, embeddings in embeddings_with_name.items()
        }

        save_embeddings(
            layer_embeddings,
            output,
            metadata=metadata,
        )
