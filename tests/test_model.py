from enum import StrEnum

from easyevo2.model import ModelType


def test_model_type_is_strenum():
    """ModelType should inherit from StrEnum."""
    assert issubclass(ModelType, StrEnum)


def test_model_type_values():
    """All expected model types should be present."""
    expected = {
        "evo2_1b_base",
        "evo2_7b_base",
        "evo2_7b",
        "evo2_7b_262k",
        "evo2_7b_microviridae",
        "evo2_20b",
        "evo2_40b_base",
        "evo2_40b",
    }
    actual = {m.value for m in ModelType}
    assert actual == expected


def test_list_models_returns_all():
    """list_models should return all model values."""
    models = ModelType.list_models()
    assert len(models) == len(ModelType)
    for m in ModelType:
        assert m.value in models


def test_new_models_present():
    """New models from evo2 v0.5.0 should be present."""
    assert hasattr(ModelType, "evo2_20b")
    assert hasattr(ModelType, "evo2_7b_262k")
    assert hasattr(ModelType, "evo2_7b_microviridae")
    assert ModelType.evo2_20b.value == "evo2_20b"
