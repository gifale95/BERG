class NESTError(Exception):
    """Base class for all NEST exceptions."""
    pass

class ModelNotFoundError(NESTError):
    """Raised when a requested model is not found."""
    pass

class ModelLoadError(NESTError):
    """Raised when a model fails to load."""
    pass

class InvalidParameterError(NESTError):
    """Raised when an invalid parameter is provided."""
    pass

class StimulusError(NESTError):
    """Raised when there's an issue with stimulus data."""
    pass

class ResourceError(NESTError):
    """Raised when there's an issue with resources (GPU, memory, etc.)."""
    pass