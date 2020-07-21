"""Provides propagators for standard Clarion subsystems."""


__all__ = ["AgentCycle", "CycleS", "ACSCycle", "NACSCycle"]


from pyClarion.base import ConstructType, Symbol, MatchSet, Cycle
from types import MappingProxyType
from typing import Dict, Mapping, Tuple


class AgentCycle(Cycle[Dict[Symbol, float], Mapping[Symbol, float]]):
    """Represents an agent activation cycle."""

    output = ConstructType.buffer | ConstructType.subsystem

    def __init__(self):

        super().__init__(
            sequence = [
                ConstructType.buffer,
                ConstructType.subsystem
            ] 
        )

    def emit(self, data: Dict[Symbol, float] = None) -> Mapping[Symbol, float]:

        mapping = data if data is not None else dict()
        return MappingProxyType(mapping=mapping)


class CycleS(Cycle[Dict[Symbol, float], Mapping[Symbol, float]]):
    """Represents a subsystem activation cycle."""

    output = ConstructType.node | ConstructType.terminus

    def __init__(self, sequence, matches: MatchSet = None):

        super().__init__(sequence=sequence, matches=matches)

    def emit(self, data: Dict[Symbol, float] = None) -> Mapping[Symbol, float]:

        mapping = data if data is not None else dict()
        return MappingProxyType(mapping=mapping)


class ACSCycle(CycleS):

    def __init__(self, matches = None):

        super().__init__(
            sequence = [
                ConstructType.flow_in,
                ConstructType.feature,
                ConstructType.flow_bt,
                ConstructType.chunk,
                ConstructType.flow_h,
                ConstructType.chunk,
                ConstructType.flow_tb,
                ConstructType.feature,
                ConstructType.terminus
            ],
            matches = matches
        )


class NACSCycle(CycleS):

    def __init__(self, matches = None):

        super().__init__(
            sequence = [
                ConstructType.flow_in,
                ConstructType.chunk,
                ConstructType.flow_tb,
                ConstructType.feature,
                ConstructType.flow_h,
                ConstructType.feature,
                ConstructType.flow_bt,
                ConstructType.chunk,
                ConstructType.terminus
            ],
            matches = matches
        )
