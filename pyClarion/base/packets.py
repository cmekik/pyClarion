"""
Tools for representing information about node activations and decisions.

This module provides classes for constructing activation and decision packets, 
which are mapping objects with node symbols as keys and activations as values. 
Additional useful metadata as to packet origin and, in decision packets, as to 
selected node(s) is also included.
"""


# Notes For Readers 

#   - Type hints signal intended usage.


import typing as typ
import pyClarion.base.symbols as sym


__all__ = [
    "ActivationPacket",
    "DecisionPacket"
]


#####################
### Type Aliases ####
#####################


DefaultActivation = typ.Callable[[typ.Optional[sym.ConstructSymbol]], typ.Any]
StrengthSequence = typ.Sequence[typ.Tuple[sym.ConstructSymbol, typ.Any]]
StrengthMapping = typ.Mapping[sym.ConstructSymbol, typ.Any]


###################
### Definitions ###
###################


class ActivationPacket(typ.Mapping[sym.ConstructSymbol, typ.Any]):
    """
    A datastructure for representing node activations and related info.
    
    It is expected that keys to `ActivationPacket` will be construct symbols 
    representing chunk or microfeature nodes.
    """

    def __init__(
        self, 
        strengths: typ.Union[StrengthMapping, StrengthSequence],
        origin: sym.ConstructSymbol = None
    ) -> None:
        '''
        Initialize an ``ActivationPacket`` instance.

        :param kvpairs: Node strengths.
        :param origin: Origin of the activation packet.
        '''

        self._mapping = dict(strengths)
        self.origin = origin

    def __iter__(self):

        return iter(self._mapping)

    def __len__(self):

        return len(self._mapping)

    def __getitem__(self, key):

        return self._mapping[key]

    def __repr__(self) -> str:
        
        return ''.join(self._repr())

    def copy(self) -> 'ActivationPacket':
        """Return a shallow copy of self."""

        return type(self)(self._mapping, self.origin)

    def _repr(self) -> typ.List[str]:

        pieces = [type(self).__name__, '(', repr(self._mapping)]
        if self.origin:
            pieces.extend((', origin=', repr(self.origin)))
        pieces.append(')')
        return pieces


class DecisionPacket(ActivationPacket):
    """
    A datastructure for representing the output of an appraisal routine.

    Contains information about selected chunks and strengths of actionable
    chunks. Keys are expected to represent chunk nodes.
    """

    def __init__(
        self, 
        strengths: typ.Union[StrengthMapping, StrengthSequence],
        chosen: typ.Sequence[sym.ConstructSymbol],
        origin: sym.ConstructSymbol = None
    ) -> None:
        '''
        Initialize a ``DecisionPacket`` instance.

        :param strengths: Strengths of actionable chunks.
        :param chosen: Selected chunks.
        :param origin: Origin of the decision.
        '''

        super().__init__(strengths, origin)
        self._chosen = chosen

    def copy(self) -> 'DecisionPacket':

        return type(self)(self._mapping, self.chosen, self.origin) 

    def _repr(self) -> typ.List[str]:

        pieces = super()._repr()
        start, end = pieces[:3], pieces[3:]
        middle = [", ", "chosen=", repr(self._chosen)]
        return start + middle + end

    @property
    def chosen(self) -> typ.Sequence[sym.ConstructSymbol]:
        """Selected chunks."""

        return self._chosen
