"""Tools for representing information about node strengths and decisions."""


from typing import Any, Mapping, Set, FrozenSet 
from pyClarion.base.symbols import ConstructSymbol


__all__ = ["Packet", "ActivationPacket", "DecisionPacket"]


class Packet(object):
    """
    Base container class facilitating communication among simulation constructs.

    Packets are the primary and, ideally, only medium of direct communication 
    among simulated Clarion constructs belonging to the same agent. 
    
    Packet objects present a dict-like API for mapping construct symbols to 
    arbitrary data structures, but, unlike dicts, they do not support in-place 
    modification. Typically, a single packet will be accessed by multiple 
    constructs. In-place modification is discouraged to prevent subtle bugs 
    arising from shared use.

    This class may be extended with additional properties if necessary, (see 
    DecisionPacket for example), but additional properties should not support 
    in-place modification: they should not have a setters and they should not 
    return mutable objects. 
    """

    __slots__ = ("_data")

    def __init__(self, data: Mapping[ConstructSymbol, Any]) -> None:
        """
        Initialize a new packet instance.

        :param data: A mapping from construct symbols to related data.
        """

        self._data = dict(data)

    def __repr__(self):

        return "<{}: {}>".format(self.__class__.__name__, self._contents_repr())

    def __contains__(self, key):

        return key in self._data

    def __iter__(self):

        return iter(self._data)

    def __len__(self):

        return len(self._data)

    def __getitem__(self, key: ConstructSymbol):

        return self._data[key]

    def get(self, key, default=None):

        return self._data.get(key, default)

    def keys(self):
        """Return a view of keys in self."""

        return self._data.keys()

    def values(self):
        """Return a view of values in self."""

        return self._data.values()

    def items(self):
        """Return a view of items in self."""

        return self._data.items()

    def _contents_repr(self):

        return "data={}".format(repr(self._data))


class ActivationPacket(Packet):
    """
    Represents chunk/feature activation strengths.
    
    See Packet documentation for details.
    """

    def __init__(self, strengths: Mapping[ConstructSymbol, Any]) -> None:

        super().__init__(strengths)


class DecisionPacket(Packet):
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
