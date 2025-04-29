from pathlib import Path

import pytest
import torch

from easyevo2.dataloader import get_seq_from_fx
from easyevo2.io import save_embeddings
from easyevo2.model import ModelType, load_model


@pytest.mark.slow
def test_evo2():
    model_type = ModelType.evo2_7b
    layer_name = ["blocks.28.mlp.l3"]
    device = "cuda:0"
    filename = "tests/data/test.fa"
    output = "tests/data/test_evo2_embeddings"

    if device.startswith("cuda") and torch.cuda.is_available():
        # Check if the specified GPU is available
        gpu_index = int(device.split(":")[1])
        if gpu_index >= torch.cuda.device_count():
            msg = f"GPU index {gpu_index} is out of range. Available GPUs: {torch.cuda.device_count()}"
            raise ValueError(msg)

    model = load_model(model_type)
    sequences = get_seq_from_fx(
        filename,
    )

    embeddings_with_name = {}

    # Process sequences in batches
    for name, seq in sequences.items():
        # Tokenize and process the sequence
        input_ids = (
            torch.tensor(
                model.tokenizer.tokenize(seq),
                dtype=torch.int,
            )
            .unsqueeze(0)
            .to(device)
        )

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
            "model_type": model_type.value,
            "layer_name": layer,
            "output": str(output),
        }

        if output is None:
            layer_output = Path(filename).with_suffix(
                f".{model_type}.{layer}.safetensors"
            )
        else:
            layer_output = Path(output).with_suffix(
                f".{model_type}.{layer}.safetensors"
            )

        layer_embeddings = {
            name: embeddings[layer].cpu()
            for name, embeddings in embeddings_with_name.items()
        }

        save_embeddings(
            layer_embeddings,
            layer_output,
            metadata=metadata,
        )
