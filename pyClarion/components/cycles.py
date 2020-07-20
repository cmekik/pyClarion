"""Provides propagators for standard Clarion subsystems."""


__all__ = ["AgentCycle", "CycleS", "ACSCycle", "NACSCycle"]


from pyClarion.base import (
    ConstructType, ConstructSymbol, Cycle, ResponsePacket, SubsystemPacket, 
    MatchSpec
)
from types import MappingProxyType
from typing import Dict, Mapping, Tuple


SPData = Tuple[
    Mapping[ConstructSymbol, Mapping[ConstructSymbol, float]],
    Mapping[ConstructSymbol, ResponsePacket]
]


class AgentCycle(Cycle[None, None]):
    """Represents an agent activation cycle."""

    def __init__(self):

        super().__init__(
            sequence = [
                ConstructType.buffer,
                ConstructType.subsystem
            ] 
        )

    def emit(self, data: None = None) -> None:
        pass


class CycleS(Cycle[Mapping[ConstructSymbol, float], Mapping[ConstructSymbol, float]]):
    """Represents a subsystem activation cycle."""

    output = ConstructType.node | ConstructType.response

    def __init__(self, sequence, matches: MatchSpec = None):

        super().__init__(sequence=sequence, matches=matches)

    def emit(
        self, data: Mapping[ConstructSymbol, float] = None
    ) -> Mapping[ConstructSymbol, float]:

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
                ConstructType.response
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
                ConstructType.response
            ],
            matches = matches
        )
