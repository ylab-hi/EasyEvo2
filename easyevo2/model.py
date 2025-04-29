from enum import Enum


class ModelType(str, Enum):
    """Enum for model types."""

    evo2_1b_base = "evo2_1b_base"  # base model with 8k context
    evo2_7b_base = "evo2_7b_base"  # base model with 8k context
    evo2_40b_base = "evo2_40b_base"  # base model with 8k context
    evo2_7b = "evo2_7b"  # 7B model with 1m context
    evo2_40b = "evo2_40b"  # 40B model with 1m context

    @classmethod
    def list_models(cls):
        """List all available model types."""
        return [model.value for model in cls]


def load_model(model_type: ModelType):
    """
    Load the specified model type.

    Args:
        model_type (str): The model type to load.

    Returns
    -------
        model: The loaded model.
    """
    if (
        model_type == ModelType.evo2_1b_base
        or model_type == ModelType.evo2_7b_base
        or model_type == ModelType.evo2_40b_base
        or model_type == ModelType.evo2_7b
        or model_type == ModelType.evo2_40b
    ):
        from evo2 import Evo2

        return Evo2(model_type.value)
    else:
        msg = f"Unknown model type: {model_type}"
        raise ValueError(msg)
