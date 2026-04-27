class StaleBackgroundJob(RuntimeError):
    """Raised when a background job targets records removed by a newer reset."""
