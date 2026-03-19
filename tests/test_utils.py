import logging

import pytest

from easyevo2.utils import log, sliding_window, validate_sequence


def test_validate_sequence_valid():
    assert validate_sequence("ACGTACGT") is True


def test_validate_sequence_lowercase():
    assert validate_sequence("acgtacgt") is True


def test_validate_sequence_invalid_chars():
    assert validate_sequence("ACGT1234") is False


def test_validate_sequence_empty():
    assert validate_sequence("") is False


def test_validate_sequence_too_long():
    assert validate_sequence("ACGT" * 100, max_length=10) is False


def test_validate_sequence_min_length():
    assert validate_sequence("A", min_length=5) is False
    assert validate_sequence("ACGTA", min_length=5) is True


def test_sliding_window_basic():
    sequences = [("seq1", "ABCDEFGH")]
    results = list(sliding_window(sequences, window_size=4, step_size=2))
    # First result is the original sequence
    assert results[0] == ("seq1", "ABCDEFGH")
    # Then windows
    assert ("seq1_0_4", "ABCD") in results
    assert ("seq1_2_6", "CDEF") in results
    assert ("seq1_4_8", "EFGH") in results


def test_sliding_window_invalid_params():
    sequences = [("seq1", "ACGT")]
    with pytest.raises(ValueError, match="window_size"):
        list(sliding_window(sequences, window_size=0, step_size=1))
    with pytest.raises(ValueError, match="step_size"):
        list(sliding_window(sequences, window_size=1, step_size=0))


def test_sliding_window_empty_sequence():
    sequences = [("seq1", "")]
    results = list(sliding_window(sequences, window_size=4, step_size=2))
    assert results == []


def test_sliding_window_short_sequence():
    sequences = [("seq1", "AB")]
    results = list(sliding_window(sequences, window_size=4, step_size=2))
    # Only the original sequence, no windows
    assert len(results) == 1
    assert results[0] == ("seq1", "AB")


def test_logger_exists():
    assert isinstance(log, logging.Logger)
    assert log.name == "easyevo2"
