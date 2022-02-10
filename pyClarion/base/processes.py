"""Basic abstractions for component processes."""


from __future__ import annotations
from typing import ClassVar, Tuple, Callable, Any
from functools import partial

from .symbols import feature


__all__ = ["Process"]


class Process:
    """Base class for simulated processes."""

    fspace_names: ClassVar = ("reprs", "cmds", "params", "flags")

    prefix: str = ""
    fspaces: Tuple[partial[Tuple[feature, ...]], ...] = ()

    initial: Any
    call: Callable

    def validate(self) -> None:
        """Validate process configuration."""
        pass

    @property
    def reprs(self) -> Tuple[feature, ...]:
        """Feature symbols for state representations."""
        raise NotImplementedError()

    @property
    def flags(self) -> Tuple[feature, ...]:
        """Feature symbols for process flags."""
        raise NotImplementedError()

    @property
    def params(self) -> Tuple[feature, ...]:
        """Feature symbols for process parameters."""
        raise NotImplementedError()

    @property
    def cmds(self) -> Tuple[feature, ...]:
        """Feature symbols for process commands."""
        raise NotImplementedError()

    @property
    def nops(self) -> Tuple[feature, ...]:
        """Feature symbols for process nop commands."""
        raise NotImplementedError()
