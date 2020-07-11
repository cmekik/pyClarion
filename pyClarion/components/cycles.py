"""Provides propagators for standard Clarion subsystems."""


__all__ = ["NACSCycle", "AgentCycle"]


from typing import Dict
from pyClarion.base import ConstructType, CycleS, CycleG

class AgentCycle(CycleG):

    def __init__(self):

        super().__init__(
            sequence = [
                ConstructType.buffer,
                ConstructType.subsystem
            ] 
        )

    def make_packet(self, data: None = None) -> None:
        pass


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
