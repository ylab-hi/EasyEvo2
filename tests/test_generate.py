import pytest
from typer import BadParameter

from easyevo2.generate import generate


def test_generate_no_input_raises():
    """Should raise when no prompt source is given."""
    with pytest.raises(BadParameter, match="exactly one"):
        generate()


def test_generate_multiple_inputs_raises():
    """Should raise when multiple prompt sources are given."""
    with pytest.raises(BadParameter, match="exactly one"):
        generate(prompt="ACGT", species="Homo sapiens")
