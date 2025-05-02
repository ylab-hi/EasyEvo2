from pathlib import Path
from typing import Annotated

import torch
import typer

from easyevo2.dataloader import get_seq_from_fx
from easyevo2.io import save_embeddings
from easyevo2.model import ModelType, load_model

# define a command-line interface (CLI) using Typer
# the cli include subcommands
app = typer.Typer(
    epilog="EasyEvo2 make life easier for you.\n",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command()
def embed(
    filename: Annotated[
        Path,
        typer.Argument(
            ...,
            help="Path to the FASTA or FASTQ file.",
        ),
    ],
    model_type: Annotated[
        ModelType,
        typer.Option(
            help="Model type to use for embedding.",
        ),
    ] = ModelType.evo2_7b,
    layer_name: Annotated[
        list[str] | None,
        typer.Option(
            help="Layer name to extract embeddings from.",
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            help="Batch size for processing sequences.",
        ),
    ] = 1,
    device: Annotated[
        str,
        typer.Option(
            help="Device to run the model on (e.g., 'cuda:0' or 'cpu').",
        ),
    ] = "cuda:0",
    output: Annotated[
        Path | None,
        typer.Option(
            help="Output file to save the embeddings.",
        ),
    ] = None,
):
    """Embed a FASTA or FASTQ file."""
    # Load the model
    if layer_name is None:
        layer_name = ["blocks.28.mlp.l3"]

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
    for name, seq in sequences:
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

            # Move embeddings to CPU to free GPU memory
            cpu_embeddings = {
                layer: tensor.detach().cpu() for layer, tensor in embeddings.items()
            }

            # Store the embeddings
            embeddings_with_name[name] = cpu_embeddings

    # Save the embeddings to the output file
    for layer in layer_name:
        metadata = {
            "model_type": model_type.value,
            "layer_name": layer,
            "batch_size": str(batch_size),
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
            name: embeddings[layer] for name, embeddings in embeddings_with_name.items()
        }

        save_embeddings(
            layer_embeddings,
            layer_output,
            metadata=metadata,
        )


@app.command()
def list_models():
    """List all available model types."""
    models = ModelType.list_models()
    for model in models:
        print(model)


if __name__ == "__main__":
    # run the CLI
    app()
