from typing import Iterator
from datetime import timedelta

from .base import Component
from ..events import Event
from ..knowledge import Family


class Simulation(Component):
    """
    A simulation environment.

    Do not directly instantiate this class. Use `Environment` or `Agent` instead.
    """

    def __init__(self, name: str, **families: Family) -> None:
        super().__init__(name)
        for name, family in families.items():
            self.system.root[name] = family

    def run(self) -> Iterator[Event]:
        """Process and yield each queued event in system."""
        while self.system.queue and self.system.clock.has_time:
            yield self.system.advance()
    
    def run_all(self) -> None:
        """Process all queued events."""
        self.system.run_all()

    def set_limit(self, limit: timedelta) -> None:
        self.system.clock.limit = limit


class Environment(Simulation):
    """Top-level process for a multi-agent simulation."""
    pass


class Agent(Simulation):
    """Top-level process for a simulated agent."""
    pass