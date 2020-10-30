"""Tools for updater setup."""


__all__ = ["UpdaterChain", "CommandDrivenUpdater"]


from pyClarion.base import Symbol, MatchSet, Updater, Realizer, Structure
from typing import Container, Tuple, TypeVar, cast, Generic


Rt = TypeVar("Rt", bound=Realizer)


class UpdaterChain(Generic[Rt]):
    """
    Wrapper for multiple updaters.

    Allows realizers to have multiple updaters which fire in a pre-defined 
    sequence. 
    """

    def __init__(self, *updaters: Updater[Rt]):
        """
        Initialize an UpdaterChain instance.
        
        :param updaters: A sequence of updaters.
        """

        self.updaters = updaters

    def __call__(self, realizer: Rt) -> None:
        """
        Update persistent information in realizer.
        
        Issues calls to member updaters in the order that they appear in 
        self.updaters.
        """

        for updater in self.updaters:
            updater(realizer)


class CommandDrivenUpdater(Generic[Rt]):
    """Conditionally issues calls to base updater based on action commands."""

    def __init__(
        self,
        controller: Tuple[Symbol, Symbol],
        conditions: Container[Symbol],
        base: Updater
    ) -> None:

        self.controller = controller
        self.conditions = conditions
        self.base = base

    def __call__(self, realizer: Rt) -> None:

        cmds = cast(Structure, realizer)[self.controller].output
        if any(cmd in self.conditions for cmd in cmds):
            self.base(realizer)
