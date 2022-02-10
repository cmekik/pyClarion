"""Basic Clarion datatypes."""


from __future__ import annotations
from typing import Union, NamedTuple


__all__ = ["dimension", "feature", "chunk", "rule"]


class dimension(NamedTuple):
    """
    A dimension symbol.
    
    :param id: Dimension URI.
    :param lag: Time-lag (in simulation steps). Defaults to 0.
    """

    id: str
    lag: int = 0


class feature(NamedTuple):
    """
    A feature symbol.
    
    :param d: Feature dimension URI.
    :param v: Feature value. If str, should be a URI.
    :param l: Feature dimension time-lag (in simulation steps). Defaults to 0.
    """
    
    d: str
    v: Union[str, int, None] = None
    l: int = 0

    @property
    def dim(self) -> dimension:
        """Feature dimension."""
        return dimension(self.d, self.l)


class chunk(NamedTuple):
    """
    A chunk symbol.
    
    :param id: Chunk URI.
    """
    
    id: str


class rule(NamedTuple):
    """
    A rule symbol.
    
    :param id: Rule URI.
    """
    
    id: str
