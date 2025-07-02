# models.py
from sklearn.semi_supervised import LabelPropagation # Keep this import

# Import the model classes from their respective files
from anfis_hybrid import HybridANFIS
from anfis_nonHyb import NoHybridANFIS
from PopFnn import POPFNN
from kmFmmc import FMNC



# A dictionary mapping string names to the actual model classes
_MODEL_CLASSES = {
    "HybridANFIS": HybridANFIS,
    "NoHybridANFIS": NoHybridANFIS,
    "POPFNN": POPFNN,
    "FMMC": FMNC,
    "LabelPropagation": LabelPropagation, # Add the sklearn model here
}

def get_model_class(model_name: str):
    """
    Retrieves a model class from the registry.

    Args:
        model_name (str): The name of the model.

    Returns:
        The uninitialized model class.
    """
    model_class = _MODEL_CLASSES.get(model_name)
    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}. Available models are: {list(_MODEL_CLASSES.keys())}")
    return model_class