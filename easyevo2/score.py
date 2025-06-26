import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from easyevo2.dataloader import get_seq_from_fx
from easyevo2.model import ModelType, load_model
from easyevo2.utils import check_cuda, sliding_window


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
