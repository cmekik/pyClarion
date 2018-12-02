"""Exception classes specific to pyClarion."""


class BasePyClarionException(Exception):
    """Base class for all exceptions specific to pyClarion."""

    pass


class ConstructError(BasePyClarionException):
    """Base class for violations of realizer expectations wrt constructs."""

    pass


class UnexpectedConstructError(ConstructError):
    """An unexpected construct was passed to a construct realizer."""

    pass


class ForbiddenConstructError(ConstructError):
    """A forbidden construct was passed to a container construct realizer."""

    pass


class ConstructMismatchError(ConstructError):
    """Container construct key does not match client construct of value."""

    pass
