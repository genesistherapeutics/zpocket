"""Local exception types for zpocket."""


class ZPocketError(RuntimeError):
    """Raised when zpocket cannot produce predictions or trunk artifacts.

    The single error type for both backends. zpocket implements no cache or retry
    semantics — on error it simply raises.
    """
