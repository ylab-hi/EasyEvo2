from easyevo2.model import ModelType


def list_models():
    """List all available model types."""
    models = ModelType.list_models()
    for model in models:
        print(model)
