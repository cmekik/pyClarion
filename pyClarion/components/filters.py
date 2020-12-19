"""Tools for filtering inputs and outputs of propagators."""


__all__ = ["Gated", "Filtered", "Pruned"]


from ..base.symbols import (
    ConstructType, Symbol, feature, subsystem, terminus
)
from .. import numdicts as nd
from ..base.components import FeatureInterface, Propagator

from itertools import product
from dataclasses import dataclass
from typing import (
    NamedTuple, Tuple, Hashable, Union, Mapping, Set, Iterable, Generic, TypeVar
)
from types import MappingProxyType
import pprint


Pt = TypeVar("Pt", bound=Propagator)


class Gated(Propagator, Generic[Pt]):
    """
    Gates output of an activation propagator.
    
    The gating function is achieved by multiplying a gating signal associated 
    with the client construct in the output of the gate by the output of the 
    base propagator.

    The gating signal is assumed to be in the interval [0, 1],
    """

    def __init__(self, base: Pt, gate: Symbol, invert: bool = False) -> None:

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
            w = 1 - w

        return w * output


class Filtered(Propagator, Generic[Pt]):
    """
    Filters the input to a propagator.
    
    Filtering is achieved by elementwise multiplication of each input to the 
    base propagator by a filtering signal from a sieve construct. 

    The filtering signal is assumed to be in the interval [0, 1].
    """

    def __init__(
        self, 
        base: Pt, 
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
            ws = 1 - ws

        preprocessed = {}
        func = self.base.expects
        expected = {src: inputs[src] for src in filter(func, inputs)}
        for src, d in expected.items():
            if src in self.exempt:
                preprocessed[src] = d
            else:
                preprocessed[src] = ws * d

        return MappingProxyType(preprocessed)


class Pruned(Propagator, Generic[Pt]):
    """
    Prunes the input to an activation propagator.
    
    Pruning is achieved by removing, in each input to the base propagator, 
    constructs of a chosen construct type. 
    """

    def __init__(
        self, 
        base: Pt, 
        accept: ConstructType,
        exempt: Set[Symbol] = None, 
        invert: bool = True
    ) -> None:

        self.base = base
        self.accept = accept
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
            ws = 1 - ws

        preprocessed = {}
        func = self.base.expects
        expected = {src: inputs[src] for src in filter(func, inputs)}
        for src, d in expected.items():
            if src in self.exempt:
                preprocessed[src] = d
            else:
                preprocessed[src] = nd.drop(d, func=self._match)

        return MappingProxyType(preprocessed)

    def _match(self, construct):

        return construct.ctype in self.sieve