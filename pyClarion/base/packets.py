"""Tools for representing information about node strengths and decisions."""


from typing import Any, Mapping, Set, FrozenSet 
from pyClarion.base.symbols import ConstructSymbol


__all__ = ["ActivationPacket", "DecisionPacket"]


class _Packet(object):

    def __init__(self, strengths: Mapping[ConstructSymbol, Any]) -> None:

        self._strengths = dict(strengths)

    def __repr__(self):

        return "<{}: {}>".format(self.__class__.__name__, self._contents_repr())

    def __contains__(self, key):

        return key in self._strengths

    def __iter__(self):

        return iter(self._strengths)

    def __len__(self):

        return len(self._strengths)

    def __getitem__(self, key: ConstructSymbol):

        return self._strengths[key]

    def get(self, key, default=None):

        return self._strengths.get(key, default)

    def keys(self):
        """Return a view of keys in self."""

        return self._strengths.keys()

    def values(self):
        """Return a view of values in self."""

        return self._strengths.values()

    def items(self):
        """Return a view of items in self."""

        return self._strengths.items()

    def _contents_repr(self):

        return "strengths={}".format(repr(self._strengths))


class ActivationPacket(_Packet):
    """
    Represents node strengths.
    
    :param strengths: Mapping of node strengths.
    """

    pass


class DecisionPacket(_Packet):
    """
    Represents a response.

    :param strengths: Mapping of node strengths.
    :param selection: Set of selected actionable nodes.
    """

    def __init__(
        self, 
        strengths: Mapping[ConstructSymbol, Any], 
        selection: Set[ConstructSymbol]
    ) -> None:

        super().__init__(strengths)
        self._selection = frozenset(selection)

    @property
    def selection(self) -> FrozenSet[ConstructSymbol]:

        return self._selection

    def _contents_repr(self):

        strengths_repr = super()._contents_repr()
        selection_repr = "selection={}".format(repr(self._selection))
        return " ".join([strengths_repr, selection_repr])
