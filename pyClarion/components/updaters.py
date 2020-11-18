"""Tools for updater setup."""


__all__ = [
    "UpdaterChain", "UpdaterChainC", "UpdaterChainS", "ConditionalUpdater", 
    "ConditionalUpdaterC", "ConditionalUpdaterS"
]


from ..base.symbols import Symbol, MatchSet, feature, subsystem, terminus
from ..base.components import (
    Propagator, Updater, UpdaterC, UpdaterS, FeatureInterface
)

from typing import Container, Tuple, TypeVar, cast, Generic, Iterable, Union
from dataclasses import dataclass
from types import MappingProxyType


Ut = TypeVar("Ut", bound="Updater")
Pt = TypeVar("Pt", bound="Propagator")
Uc_t = TypeVar("Uc_t", bound="UpdaterC")
Us_t = TypeVar("Us_t", bound="UpdaterS")


class UpdaterChain(Updater, Generic[Ut]):
    """
    Wrapper for multiple updaters.

    Allows realizers to have multiple updaters which fire in a pre-defined 
    sequence. 
    """

    updaters: Tuple[Ut, ...]

    def __init__(self, *updaters: Ut):

        self.updaters = updaters

    def entrust(self, construct):

        for updater in self.updaters:
            updater.entrust(construct)
        self._client = construct

    def expects(self, construct):

        return any([updater.expects(construct) for updater in self.updaters])

    def _extract_update_data(self, updater, update_data):
        
        items = update_data.items()
        extracted = {src: data for src, data in items if updater.expects(src)}
        update_data_proxy = MappingProxyType(extracted)

        return update_data_proxy


class UpdaterChainC(UpdaterChain[Uc_t], UpdaterC[Pt], Generic[Pt, Uc_t]):
    """Updater chain for basic constructs."""

    def __call__(self, emitter, inputs, output, update_data) -> None:
        """
        Update persistent information in realizer.
        
        Issues calls to member updaters in the order that they appear in 
        self.updaters.
        """

        for updater in self.updaters:
            update_data_proxy = self._extract_update_data(updater, update_data)
            updater(emitter, inputs, output, update_data_proxy)


class UpdaterChainS(UpdaterChain[Us_t], UpdaterS, Generic[Us_t]):
    """Updater chain for composite constructs (i.e. structures)."""

    def __call__(self, inputs, output, update_data) -> None:
        """
        Update persistent information in realizer.
        
        Issues calls to member updaters in the order that they appear in 
        self.updaters.
        """

        for updater in self.updaters:
            update_data_proxy = self._extract_update_data(updater, update_data)
            updater(inputs, output, update_data_proxy)


class ConditionalUpdater(Updater, Generic[Ut]):
    """Conditionally issues calls to base updater based on action commands."""

    interface: "Interface"

    @dataclass
    class Interface(FeatureInterface):
        
        has_defaults = False

        conditions: Iterable[feature] 

        def _set_interface_properties(self):

            self._features = frozenset(self.conditions)
            self._defaults = frozenset()
            self._tags = frozenset(f.tag for f in self._features)
            self._dims = frozenset(f.dim for f in self._features)

        def _validate_data(self):
            
            pass

    def __init__(
        self,
        controller: Union[Tuple[subsystem, terminus], terminus],
        interface: Interface,
        base: Ut
    ) -> None:

        self.controller = controller
        self.interface = interface
        self.base = base

    def entrust(self, construct):

        for updater in self.updaters:
            updater.entrust(construct)
        self._client = construct

    def expects(self, construct):

        val = self.base.expects(construct)
        if not isinstance(self.controller, terminus):
            subsystem, _ = self.controller
            val |= construct == subsystem 
        
        return val

    def _extract_cmds(self, output):

        if isinstance(self.controller, terminus):
            cmds = output[self.controller]
        else:
            _subsystem, _terminus = self.controller
            cmds = output[_subsystem][_terminus]

        return cmds


class ConditionalUpdaterC(
    ConditionalUpdater[Uc_t], UpdaterC[Pt], Generic[Pt, Uc_t]
):
    """Conditional updater for basic constructs."""

    def __call__(self, emitter, inputs, output, update_data) -> None:

        cmds = self._extract_cmds(output)
        if any(cmd in self.interface.features for cmd in cmds):
            self.base(emitter, inputs, output, update_data)


class ConditionalUpdaterS(
    ConditionalUpdater[Us_t], UpdaterS, Generic[Pt, Us_t]
):
    """Conditional updater for composite constructs (i.e., structures)."""

    def __call__(self, inputs, output, update_data) -> None:

        cmds = self._extract_cmds(output)
        if any(cmd in self.interface.features for cmd in cmds):
            self.base(inputs, output, update_data)
