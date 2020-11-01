"""Provides propagators for standard Clarion subsystems."""


__all__ = ["AgentCycle", "CycleS", "ACSCycle", "NACSCycle"]


from ..base import ConstructType, Symbol, MatchSet, Cycle
from types import MappingProxyType
from typing import Dict, Mapping, Tuple, Container


class AgentCycle(Cycle):
    """Represents an agent activation cycle."""

    _serves = ConstructType.agent
    output = ConstructType.buffer | ConstructType.subsystem
    sequence = [
        ConstructType.buffer,
        ConstructType.subsystem
    ]

    def __init__(self):
       
        self.sequence = type(self).sequence

    def expects(self, construct: Symbol):

        return False

    @staticmethod
    def emit(data: Dict[Symbol, float] = None) -> Mapping[Symbol, float]:

        mapping = data if data is not None else dict()
        return MappingProxyType(mapping=mapping)


class CycleS(Cycle):
    """Represents a subsystem activation cycle."""

    _serves = ConstructType.subsystem
    # NOTE: Should flows be added to output? - Can
    output = ConstructType.nodes | ConstructType.terminus

    def __init__(self, sources: Container[Symbol] = None):

        self.sources = sources if sources is not None else set()
        self.sequence = type(self).sequence

    def expects(self, construct: Symbol):

        return construct in self.sources

    @staticmethod
    def emit(data: Dict[Symbol, float] = None) -> Mapping[Symbol, float]:

        mapping = data if data is not None else dict()
        return MappingProxyType(mapping=mapping)


class ACSCycle(CycleS):

    sequence = [
        ConstructType.flow_in,
        ConstructType.features,
        ConstructType.flow_bt,
        ConstructType.chunks,
        ConstructType.flow_h,
        ConstructType.chunks,
        ConstructType.flow_tb,
        ConstructType.features,
        ConstructType.terminus
    ]


class NACSCycle(CycleS):

    sequence = [
        ConstructType.flow_in,
        ConstructType.chunks,
        ConstructType.flow_tb,
        ConstructType.features,
        ConstructType.flow_h,
        ConstructType.features,
        ConstructType.flow_bt,
        ConstructType.chunks,
        ConstructType.terminus
    ]

