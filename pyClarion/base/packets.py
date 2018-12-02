"""Tools for representing information about node activations and decisions."""


###############
### IMPORTS ###
###############


import typing as typ
import pyClarion.base.symbols as sym


#####################
### TYPE ALIASES ####
#####################


At = typ.TypeVar("At")
DefaultActivation = typ.Callable[[typ.Optional[sym.Node]], At]


###################
### DEFINITIONS ###
###################


class ActivationPacket(dict, typ.MutableMapping[sym.Node, At]):
    """Represents node activations."""

    def __init__(
        self, 
        kvpairs: typ.Union[
            typ.Mapping[sym.Node, At], typ.Sequence[typ.Tuple[sym.Node, At]]
        ] = None,
        origin: typ.Hashable = None
    ) -> None:
        '''
        Initialize an ``ActivationPacket`` instance.

        :param kvpairs: Node strengths.
        :param origin: Contains necessary information about the origin of the 
            activation pattern.
        '''

        super().__init__()
        if kvpairs:
            self.update(kvpairs)
        self.origin = origin

    def __repr__(self) -> str:
        
        return ''.join(self._repr())

    def copy(self):
        """Return a shallow copy of self."""

        return self.subpacket(self.keys())

    def subpacket(
        self, 
        nodes: typ.Iterable[sym.Node], 
        default_activation: DefaultActivation = None
    ) -> 'ActivationPacket[At]':
        """
        Return a subpacket containing activations for chosen nodes.

        :param nodes: Keys for constructed subpacket.
        :default_activation: Procedure for determining default activations. Used
            to set activations in output packet if `nodes` contains elements not 
            contained in self.
        """
        
        mapping: dict = self._subpacket(nodes, default_activation)
        origin = self.origin
        output: 'ActivationPacket[At]' = ActivationPacket(mapping, origin)
        return output

    def _repr(self) -> typ.List[str]:

        repr_ = [
            type(self).__name__,
            '(',
            super().__repr__(),
            ", ",
            "origin=",
            repr(self.origin),
            ")"
        ]
        return repr_

    def _subpacket(
        self, 
        nodes: typ.Iterable[sym.Node], 
        default_activation: DefaultActivation = None
    ) -> dict:

        output: dict = {}
        for node in nodes:
            if node in self:
                activation = self[node]
            elif default_activation:
                activation = default_activation(node)
            else:
                raise KeyError(
                    "Node {} not in self".format(str(node))
                )
            output[node] = activation
        return output


class DecisionPacket(ActivationPacket[At]):
    """
    Represents the output of an action selection routine.

    Contains information about the selected actions and strengths of actionable
    chunks.
    """

    def __init__(
        self, 
        kvpairs: typ.Mapping[sym.Node, At] = None,
        origin: typ.Hashable = None,
        chosen: typ.Set[sym.Chunk] = None
    ) -> None:
        '''
        Initialize a ``DecisionPacket`` instance.

        :param kvpairs: Strengths of actionable chunks.
        :param chosen: The set of actions to be fired.
        '''

        super().__init__(kvpairs, origin)
        self.chosen = chosen

    def __eq__(self, other: typ.Any) -> bool:

        return (
            super().__eq__(other) and
            self.chosen == other.chosen
        )

    def _repr(self) -> typ.List[str]:

        repr_ = super()._repr()
        supplement = [
            ", ",
            "chosen=",
            repr(self.chosen),
        ]
        return repr_[:-1] + supplement + repr_[-1:]

    def subpacket(
        self, 
        nodes: typ.Iterable[sym.Node], 
        default_activation: DefaultActivation = None
    ) -> 'DecisionPacket[At]':
        
        mapping = self._subpacket(nodes, default_activation)
        origin = self.origin
        chosen = self.chosen
        output: 'DecisionPacket[At]' = DecisionPacket(
            mapping, origin, chosen
        )
        return output
        