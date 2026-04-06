from enum import StrEnum
from typing import Any


class ModelType(StrEnum):
    """Enum for supported Evo2 model types."""

    evo2_1b_base = "evo2_1b_base"  # 1B params, 8k context, FP8 required
    evo2_7b_base = "evo2_7b_base"  # 7B params, 8k context
    evo2_7b = "evo2_7b"  # 7B params, 1M context
    evo2_7b_262k = "evo2_7b_262k"  # 7B params, 262k context
    evo2_7b_microviridae = "evo2_7b_microviridae"  # 7B fine-tuned on Microviridae
    evo2_20b = "evo2_20b"  # 20B params, 1M context, FP8 required
    evo2_40b_base = "evo2_40b_base"  # 40B params, 8k context, FP8 required
    evo2_40b = "evo2_40b"  # 40B params, 1M context, FP8 required

    @classmethod
    def list_models(cls) -> list[str]:
        """List all available model types."""
        return [model.value for model in cls]


def load_model(model_type: ModelType) -> Any:
    """
    Load an Evo2 model by type.

    Parameters
    ----------
    model_type : ModelType
        The model variant to load.

    Returns
    -------
    Any
        The loaded Evo2 model instance.
    """
    from evo2 import Evo2

    return Evo2(model_type.value)
