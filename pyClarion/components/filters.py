"""Tools for filtering inputs and outputs of propagators."""


__all__ = ["Gated", "Filtered"]


from ..base.symbols import (
    ConstructType, Symbol, feature, subsystem, terminus
)
from ..base import numdicts as nd
from ..base.components import FeatureInterface, Propagator

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

    @property
    def expected(self):

        return self.base.expected.union((self.gate,))

    def call(self, inputs):

        preprocessed = self.preprocess(inputs)
        strengths = self.base.call(preprocessed)
        d = self.postprocess(inputs, strengths)

        return d

    def update(self, inputs, output):
        """
        Call update routine for base.

        Updates to base may behave strangely due to unexpected output values 
        (base will not know that gating has occurred).
        """

        # May need to add an optional `mask` arg to propagator.update() to 
        # ensure updates are computed correctly under output gating. - Can

        preprocessed = self.preprocess(inputs)
        self.base.update(preprocessed, output)

    def preprocess(self, inputs):

        func = self.base.expects
        expected = {src: inputs[src] for src in filter(func, inputs)}
    
        return MappingProxyType(expected)

    def postprocess(self, inputs, output):

        w = inputs[self.gate][self.client]
        if self.invert:
            w = 1.0 - w

        return w * output


class Filtered(Propagator):
    """Filters input to a propagator."""
    
    def __init__(
        self, 
        base: Propagator, 
        sieve: Symbol,
        exempt: Set[Symbol] = None, 
        invert: bool = True
    ) -> None:

        self.base = base
        self.sieve = sieve
        self.exempt = exempt or set() 
        self.invert = invert

    @property
    def client(self):

        return self.base.client

    def entrust(self, construct):

        self.base.entrust(construct)

    @property
    def expected(self):

        return self.base.expected.union((self.sieve,))

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

        preprocessed = {}
        func = self.base.expects
        expected = {src: inputs[src] for src in filter(func, inputs)}
        for src, d in expected.items():
            if src in self.exempt:
                preprocessed[src] = d
            else:
                preprocessed[src] = ws * d

        return MappingProxyType(preprocessed)
