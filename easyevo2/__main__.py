import json
import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import torch
import typer
from rich.logging import RichHandler
from rich.progress import track

from easyevo2.dataloader import get_seq_from_fx
from easyevo2.io import save_embeddings
from easyevo2.model import ModelType, load_model
from easyevo2.utils import check_cuda, sliding_window

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger("rich")

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
    progress: Annotated[
        bool,
        typer.Option(
            help="Whether to show a progress bar.",
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            help="Output file to save the embeddings.",
        ),
    ] = None,
    save_interval: Annotated[
        int,
        typer.Option(
            help="Save embeddings every N sequences processed.",
        ),
    ] = 10,
    max_retries: Annotated[
        int,
        typer.Option(
            help="Maximum number of retries for failed sequences.",
        ),
    ] = 3,
):
    """Embed a FASTA or FASTQ file with improved error handling and early saving."""
    # Load the model
    if layer_name is None:
        layer_name = ["blocks.28.mlp.l3"]

    check_cuda(device)

    model = load_model(model_type)
    sequences = list(get_seq_from_fx(filename))

    # Initialize tracking variables
    embeddings_with_name = {}
    failed_sequences = []
    successful_count = 0
    failed_count = 0

    # Create output paths for each layer
    layer_outputs = {}
    for layer in layer_name:
        if output is None:
            layer_output = Path(filename).with_suffix(
                f".{model_type}.{layer}.safetensors"
            )
        else:
            layer_output = Path(output).with_suffix(
                f".{model_type}.{layer}.safetensors"
            )
        layer_outputs[layer] = layer_output

    def save_current_embeddings():
        """Save current embeddings to files."""
        for layer in layer_name:
            if not embeddings_with_name:  # Skip if no embeddings
                continue

            metadata = {
                "model_type": model_type.value,
                "layer_name": layer,
                "batch_size": str(batch_size),
                "output": str(layer_outputs[layer]),
                "successful_count": str(successful_count),
                "failed_count": str(failed_count),
                "total_processed": str(successful_count + failed_count),
            }

            layer_embeddings = {
                name: embeddings[layer]
                for name, embeddings in embeddings_with_name.items()
                if layer in embeddings
            }

            if layer_embeddings:  # Only save if we have embeddings for this layer
                save_embeddings(
                    layer_embeddings,
                    layer_outputs[layer],
                    metadata=metadata,
                )

    # Process sequences one by one (since Evo2 can only handle single sequences)
    for i, seq_data in enumerate(
        track(sequences, description="Embedding sequences", disable=not progress)
    ):
        name = seq_data[0]
        seq = seq_data[1]

        # Skip if sequence is empty
        if not seq or len(seq.strip()) == 0:
            failed_sequences.append((name, "Empty sequence"))
            failed_count += 1
            continue

        success = False
        retry_count = 0

        while not success and retry_count < max_retries:
            try:
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
                        layer: tensor.detach().cpu()
                        for layer, tensor in embeddings.items()
                    }

                    # Store the embeddings
                    embeddings_with_name[name] = cpu_embeddings

                    success = True
                    successful_count += 1

                # Clear GPU memory
                del input_ids, outputs, embeddings
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

            except Exception as e:
                retry_count += 1
                error_msg = f"Attempt {retry_count}: {e}"

                if retry_count >= max_retries:
                    failed_sequences.append((name, error_msg))
                    failed_count += 1
                    print(
                        f"Failed to process sequence '{name}' after {max_retries} attempts: {e}"
                    )
                else:
                    print(
                        f"Retrying sequence '{name}' (attempt {retry_count + 1}/{max_retries}): {e}"
                    )

                # Clear any partial results
                if name in embeddings_with_name:
                    del embeddings_with_name[name]

                # Clear GPU memory on error
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

        # Save embeddings periodically
        if (i + 1) % save_interval == 0:
            save_current_embeddings()
            if progress:
                print(f"Saved {successful_count} embeddings, {failed_count} failed")

    # Save final embeddings
    save_current_embeddings()

    # Print summary
    log.info("\nProcessing complete!")
    log.info(f"Successfully processed: {successful_count} sequences")
    log.info(f"Failed: {failed_count} sequences")

    if failed_sequences:
        log.info("\nFailed sequences:")
        for name, error in failed_sequences[:10]:  # Show first 10 failures
            log.info(f"  {name}: {error}")
        if len(failed_sequences) > 10:
            log.info(f"  ... and {len(failed_sequences) - 10} more")

        # Save failed sequences to a file for reference
        failed_file = Path(filename).with_suffix(".failed_sequences.txt")
        with failed_file.open("w") as f:
            f.write("Failed sequences:\n")
            for name, error in failed_sequences:
                f.write(f"{name}\t{error}\n")
        log.info(f"Failed sequences saved to: {failed_file}")

    # Print output file locations
    for layer in layer_name:
        log.info(f"Layer '{layer}' embeddings saved to: {layer_outputs[layer]}")


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
    save_sequence: Annotated[
        bool,
        typer.Option(
            help="Whether to save the windows sequences as a separate file.",
        ),
    ] = False,
    sequence_without_windows: Annotated[
        bool,
        typer.Option(
            help="Whether to use sequence without windows as windows sequence for scoring.",
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
            sliding_window(
                sequences,
                window_size,
                step_size,
                use_sequence_without_windows=sequence_without_windows,
            )
        )

        # Create DataFrame for efficient processing
        df = pd.DataFrame(
            sliding_window_sequences, columns=["sequence_name", "sequence"]
        )

        if save_sequence:
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
        df = df.drop(columns=["sequence"])

        # Prepare output path
        if output is None:
            output = Path(filename).with_suffix(
                f".probs_{model_type}_{window_size}_{step_size}.csv"
            )

        # Save results with metadata
        metadata = {
            "model_type": model_type.value,
            "sequence_without_windows": sequence_without_windows,
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
