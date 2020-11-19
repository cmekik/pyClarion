"""Tools for filtering inputs and outputs of propagators."""


__all__ = ["Gated", "Filtered"]


from ..base.symbols import (
    Symbol, ConstructType, feature, subsystem, terminus
)
from ..base import numdicts as nd
from ..base.components import FeatureInterface, Propagator
from ..utils.funcs import collect_cmd_data

from itertools import product
from dataclasses import dataclass
from typing import NamedTuple, Tuple, Hashable, Union, Mapping, Set, Iterable
from types import MappingProxyType
import pprint


class Gated(Propagator):
    """Gates output of an activation propagator."""
    
    def __init__(
        self, 
        base: Propagator, 
        gate: Symbol,
        invert: bool = False
    ) -> None:

        self.base = base
        self.gate = gate
        self.invert = invert

    @property
    def client(self):

        return self.base.client

    def entrust(self, construct):

        self.base.entrust(construct)

    def expects(self, construct):

        return construct == self.gate or self.base.expects(construct)

    def call(self, inputs):

        w = inputs[self.gate][self.client]
        if self.invert:
            w = 1.0 - w

        func = self.base.expects
        expected = {src: inputs[src] for src in filter(func, inputs)}
        strengths = self.base.call(MappingProxyType(expected))

        return w * strengths


class Filtered(Propagator):
    """Filters input to a terminus."""
    
    def __init__(
        self, 
        base: Propagator, 
        sieve: Symbol, 
        invert: bool = True
    ) -> None:

        self.base = base
        self.sieve = sieve
        self.invert = invert

    @property
    def client(self):

        return self.base.client

    def entrust(self, construct):

        self.base.entrust(construct)

    def expects(self, construct):

        return construct == self.sieve or self.base.expects(construct)

    def call(self, inputs):

        preprocessed = self.preprocess(inputs)
        output = self.base.call(preprocessed)

        return output

    def update(self, inputs, output):

        preprocessed = self.preprocess(inputs)
        self.base.update(preprocessed, output)

    def preprocess(self, inputs):

        ws = inputs[self.sieve]
        if self.invert:
            ws = ~ ws

        func = self.base.expects
        expected = {src: inputs[src] for src in filter(func, inputs)}
        items = expected.items()
        preprocessed = {src: ws * d for src, d in items} 

        return MappingProxyType(preprocessed)
