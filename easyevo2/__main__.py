import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import torch
import typer

from easyevo2.dataloader import get_seq_from_fx
from easyevo2.io import save_embeddings
from easyevo2.model import ModelType, load_model
from easyevo2.utils import check_cuda, sliding_window

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

    check_cuda(device)

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


@app.command()
def score(
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
    ],
    window_size: Annotated[
        int,
        typer.Option(
            help="Window size for calculating probabilities.",
        ),
    ],
    step_size: Annotated[
        int,
        typer.Option(
            help="Step size for calculating probabilities.",
        ),
    ],
    device: Annotated[
        str,
        typer.Option(
            help="Device to run the model on (e.g., 'cuda:0' or 'cpu').",
        ),
    ] = "cuda:0",
    sequence: Annotated[
        bool,
        typer.Option(
            help="Whether to store single sequence.",
        ),
    ] = False,
    save_windows: Annotated[
        bool,
        typer.Option(
            help="Whether to save the windows sequences as a separate file.",
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            help="Output file path for probabilities. If not specified, will use input filename with .probs.csv suffix.",
        ),
    ] = None,
):
    """Calculate probabilities for a FASTA or FASTQ file."""
    try:
        check_cuda(device)

        # Load model and sequences
        model = load_model(model_type)
        sequences = get_seq_from_fx(filename)

        # Process sequences in sliding windows
        sliding_window_sequences = list(
            sliding_window(sequences, window_size, step_size)
        )

        # Create DataFrame for efficient processing
        df = pd.DataFrame(
            sliding_window_sequences, columns=["sequence_name", "sequence"]
        )

        if save_windows:
            output_filename = Path(filename).with_suffix(
                f".windows_{window_size}_{step_size}.fa"
            )
            print(
                f"Saving {len(sliding_window_sequences)} windows to {output_filename}"
            )
            # save the windows sequences to a fasta file
            with output_filename.open("w") as f:
                for seq in sliding_window_sequences:
                    f.write(f">{seq[0]}\n{seq[1]}\n")

        # Calculate probabilities in batches
        probs = model.score_sequences(df["sequence"].tolist())

        df["probability"] = probs

        if not sequence:
            df = df.drop(columns=["sequence"])

        # Prepare output path
        if output is None:
            output = Path(filename).with_suffix(
                f".probs_{model_type}_{window_size}_{step_size}.csv"
            )

        # Save results with metadata
        metadata = {
            "model_type": model_type.value,
            "window_size": window_size,
            "step_size": step_size,
            "device": device,
            "timestamp": pd.Timestamp.now().isoformat(),
            "output": str(output),
        }

        # Save to CSV with metadata
        df.to_csv(output, index=False)

        # Save metadata to a separate JSON file
        metadata_path = output.with_suffix(".metadata.json")
        with Path(metadata_path).open("w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Results saved to {output}")
        print(f"Metadata saved to {metadata_path}")

    except Exception as e:
        print(f"Error processing file: {e}")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    # run the CLI
    app()
