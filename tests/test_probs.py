import json
from pathlib import Path

import pandas as pd

from easyevo2.dataloader import get_seq_from_fx
from easyevo2.utils import sliding_window


def fake_probs(seqs: list[str]) -> list[float]:
    return [0.5] * len(seqs)


def test_probs():
    filename = "tests/data/test.fa"
    sliding_window_size = 100
    step_size = 10
    output = None

    sequences = get_seq_from_fx(
        filename,
    )

    sliding_window_sequences = sliding_window(sequences, sliding_window_size, step_size)

    # Create DataFrame for efficient processing
    df = pd.DataFrame(sliding_window_sequences, columns=["sequence_name", "sequence"])

    probs = fake_probs(df["sequence"].tolist())

    assert len(df.columns) == 2

    df["probability"] = probs

    # Prepare output path
    if output is None:
        output = Path(filename).with_suffix(".probs.csv")

    # Save results with metadata
    metadata = {
        "model_type": "fake",
        "window_size": sliding_window_size,
        "step_size": step_size,
        "device": "cpu",
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    # Save to CSV with metadata
    df.to_csv(output, index=False)

    # Save metadata to a separate JSON file
    metadata_path = output.with_suffix(".metadata.json")
    with Path(metadata_path).open("w") as f:
        json.dump(metadata, f, indent=2)

    assert output.exists()
    assert metadata_path.exists()

    # remove the output and metadata
    output.unlink()
    metadata_path.unlink()
