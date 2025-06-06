import pandas as pd

from easyevo2.dataloader import get_seq_from_fx
from easyevo2.utils import sliding_window


def test_probs():
    filename = "tests/data/test.fa"
    sliding_window_size = 100
    step_size = 10

    sequences = get_seq_from_fx(
        filename,
    )

    sliding_window_sequences = sliding_window(sequences, sliding_window_size, step_size)

    # Create DataFrame for efficient processing
    df = pd.DataFrame(sliding_window_sequences, columns=["sequence_name", "sequence"])

    assert len(df.columns) == 2
