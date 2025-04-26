from enum import Enum
from pathlib import Path
from typing import Annotated

import torch
import typer
from evo2 import Evo2

# define a command-line interface (CLI) using Typer
# the cli include subcommands
app = typer.Typer(
    epilog="EasyEvo2 make life easier for you.\n",
    context_settings={"help_option_names": ["-h", "--help"]},
)


class ModelType(str, Enum):
    """Enum for model types."""

    # Define the model types
    _1b = "1b"
    _7b = "7b"
    _40b = "40b"


@app.command()
def embed(
    filename=Annotated[
        Path,
        typer.Argument(
            ...,
            help="Path to the FASTA or FASTQ file.",
        ),
    ],
    model=Annotated[
        ModelType,
        typer.Option(
            ModelType._1b,
            help="Model type to use for embedding.",
        ),
    ],
    layer_name=Annotated[
        str,
        typer.Option(
            "blocks.28.mlp.l3",
            help="Layer name to extract embeddings from.",
        ),
    ],
    batch_size=Annotated[
        int,
        typer.Option(
            32,
            help="Batch size for processing sequences.",
        ),
    ],
    device=Annotated[
        str,
        typer.Option(
            "cuda:0",
            help="Device to run the model on (e.g., 'cuda:0' or 'cpu').",
        ),
    ],
    max_length=Annotated[
        int,
        typer.Option(
            2000,
            help="Maximum sequence length to process.",
        ),
    ],
    output=Annotated[
        Path,
        typer.Option(
            None,
            help="Output file to save the embeddings.",
        ),
    ],
):
    """Embed a FASTA or FASTQ file."""
    # Load the model
    device = torch.device(device)
    model = Evo2(model).to(device)

    # Read sequences from the file
    sequences = get_seq_from_fx(filename)

    # Process sequences in batches
    for batch in sequences:
        # Tokenize and process the sequence
        input_ids = (
            torch.tensor(
                model.tokenizer.tokenize(batch),
                dtype=torch.int,
            )
            .unsqueeze(0)
            .to(device)
        )

        # Get embeddings
        outputs, embeddings = model(
            input_ids, return_embeddings=True, layer_names=[layer_name]
        )

        # Save embeddings to output file if specified
        if output:
            with open(output, "a") as f:
                f.write(f"{batch}\t{embeddings[layer_name].cpu().numpy()}\n")


if __name__ == "__main__":
    # run the CLI
    app()
