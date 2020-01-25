"""Tools for representing information about node strengths and decisions."""


from typing import Any, Mapping, Set, FrozenSet, Tuple
from types import MappingProxyType 
from pyClarion.base.symbols import ConstructSymbol
from pyClarion.utils.str_funcs import pstr_iterable, pstr_iterable_cb


__all__ = ["Packet", "ActivationPacket", "DecisionPacket", "SubsystemPacket"]


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
    _format = {"indent": 4, "digits": 3}

    def __init__(self, data: Mapping[ConstructSymbol, Any] = None) -> None:
        """
        Initialize a new packet instance.

        :param data: A mapping from construct symbols to related data.
        """

        self._data = dict(data) if data is not None else dict()

    def __repr__(self):

        return "{}({})".format(self.__class__.__name__, self._contents_repr())

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

    def pstr(self):
        """Pretty print packet contents for inspection and reporting."""

        main = type(self).__name__ + "(\n{content}\n)"
        attr_str = ",\n".join(self._attrs_to_strs())
        return main.format(content=attr_str)

    def _contents_repr(self) -> str:

        return "data={}".format(repr(self._data))

    def _attrs_to_strs(self) -> Tuple[str, ...]:

        delim = "{outer}strengths = {content}" 
        content = pstr_iterable(
            iterable=self._data, 
            cb=pstr_iterable_cb, 
            cbargs={"digits": self._format["digits"]},
            indent=self._format["indent"],
            level=1
        )
        strength_pstr = delim.format(
            outer=" "*self._format["indent"], 
            content=content
        )
        
        return strength_pstr,


class ActivationPacket(Packet):
    """
    Represents chunk/feature activation strengths.
    
    See Packet documentation for details.
    """

    def __init__(self, strengths: Mapping[ConstructSymbol, Any] = None) -> None:

        super().__init__(strengths)

    def _contents_repr(self):

        return "strengths={}".format(repr(self._data))


class DecisionPacket(Packet):
    """
    Represents a response.

    :param strengths: Mapping of node strengths.
    :param selection: Set of selected actionable nodes.
    """

    def __init__(
        self, 
        strengths: Mapping[ConstructSymbol, Any] = None, 
        selection: Set[ConstructSymbol] = None
    ) -> None:

        super().__init__(strengths)
        self._selection = (
            frozenset(selection) if selection is not None else frozenset()
        )

    @property
    def selection(self) -> FrozenSet[ConstructSymbol]:

        return self._selection

    def _contents_repr(self):

        strengths_repr = "strengths={}".format(repr(self._data))
        selection_repr = "selection={}".format(repr(self._selection))
        return ", ".join([strengths_repr, selection_repr])

    def _attrs_to_strs(self):

        attr_strs = super()._attrs_to_strs()

        delim = "{outer}selection = {content}" 
        content = pstr_iterable(
            iterable=self._selection, 
            cb=pstr_iterable_cb, 
            cbargs={"digits": self._format["digits"]},
            indent=self._format["indent"],
            level=1
        )
        selection_pstr = delim.format(
            outer=" "*self._format["indent"], 
            content=content
        )        

        return attr_strs + (selection_pstr,)

class SubsystemPacket(Packet):
    """
    Represents the current state of a subsystem.
    """

    def __init__(
        self, 
        strengths: Mapping[ConstructSymbol, Any] = None, 
        decisions: Mapping[ConstructSymbol, DecisionPacket] = None
    ) -> None:

        super().__init__(strengths)
        self._decisions = (
            decisions if decisions is not None else dict()
        )

    @property
    def decisions(self) -> Mapping[ConstructSymbol, DecisionPacket]:

        # Dev Note:
        # Type annotation not happy for some reason. MappingProxyType not 
        # considered a mapping? -CSM
        return MappingProxyType(self._decisions)

    def _contents_repr(self):

        strengths_repr = "strengths={}".format(repr(self._data))
        selection_repr = "decisions={}".format(repr(self._decisions))
        return ", ".join([strengths_repr, selection_repr])

    def _attrs_to_strs(self):

        attr_strs = super()._attrs_to_strs()

        delim = "{outer}decisions = {content}" 
        content = pstr_iterable(
            iterable=self._decisions, 
            cb=self._pstr_cb, 
            cbargs={
                "indent": self._format["indent"],
                "level": 2,
                "digits": self._format["digits"]
            },
            indent=self._format["indent"],
            level=1
        )
        decisions_pstr = delim.format(
            outer=" "*self._format["indent"], 
            content=content
        )        

        return attr_strs + (decisions_pstr,)

    @staticmethod
    def _pstr_cb(obj, indent=4, level=0, digits=None):

        if isinstance(obj, DecisionPacket):
            indent_str = " " * indent * level
            s = obj.pstr().replace("\n", "\n" + indent_str)
            return s
        else:
            return pstr_iterable_cb(obj, digits)
