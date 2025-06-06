class BERGError(Exception):
    """Base class for all BERG exceptions."""
    pass

class ModelNotFoundError(BERGError):
    """Raised when a requested model is not found."""
    pass

class ModelLoadError(BERGError):
    """Raised when a model fails to load."""
    pass

class InvalidParameterError(BERGError):
    """Raised when an invalid parameter is provided."""
    pass

class StimulusError(BERGError):
    """Raised when there's an issue with stimulus data."""
    pass

class ResourceError(BERGError):
    """Raised when there's an issue with resources (GPU, memory, etc.)."""
    pass