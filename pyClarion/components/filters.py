"""Tools for filtering inputs and outputs of propagators."""


__all__ = ["Gated", "Filtered", "Pruned"]


from ..base.symbols import (
    ConstructType, Symbol, SymbolicAddress, 
    feature, subsystem, terminus
)
from ..base.components import WrappedProcess, Pt, FeatureInterface
from .. import numdicts as nd

from itertools import product
from dataclasses import dataclass
from typing import (
    NamedTuple, Tuple, Hashable, Union, Mapping, Set, Iterable, Generic, TypeVar
)
from types import MappingProxyType
import pprint


class Gated(WrappedProcess[Pt]):
    """
    Gates output of an activation propagator.
    
    The gating function is achieved by multiplying a gating signal associated 
    with the client construct in the output of the gate by the output of the 
    base propagator.

    The gating signal is assumed to be in the interval [0, 1],
    """

    def __init__(self, base: Pt, gate: Symbol, invert: bool = False) -> None:

        super().__init__(base=base, expected=(gate,))
        self.invert = invert

    def postprocess(
        self, 
        inputs: Mapping[Tuple[Symbol, ...], nd.NumDict], 
        output: nd.NumDict
    ) -> nd.NumDict:

        data, = self.extract_inputs(inputs)[:len(self.expected_top)]
        # TODO: Weight extraction should be based on full addresses? - Can
        w = data[self.client[-1]] 
        if self.invert:
            w = 1 - w

        return w * output


class Filtered(WrappedProcess[Pt]):
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
        exempt: Set[SymbolicAddress] = None, 
        invert: bool = True
    ) -> None:

        super().__init__(base=base, expected=(sieve,))
        self.exempt = exempt or set() 
        self.invert = invert

    def preprocess(self, inputs):

        ws, = self.extract_inputs(inputs)[:len(self.expected_top)]
        if self.invert:
            ws = 1 - ws

        preprocessed = {}
        for source in self.base.expected:
            if source in self.exempt:
                preprocessed[source] = inputs[source]
            else:
                preprocessed[source] = ws * inputs[source]

        return MappingProxyType(preprocessed)


class Pruned(WrappedProcess[Pt]):
    """
    Prunes the input to an activation propagator.
    
    Pruning is achieved by removing, in each input to the base propagator, 
    constructs of a chosen construct type. 
    """

    def __init__(
        self, 
        base: Pt, 
        accept: ConstructType,
        exempt: Set[SymbolicAddress] = None
    ) -> None:

        super().__init__(base=base)
        self.accept = accept
        self.exempt = exempt or set() 

    def preprocess(self, inputs):

        preprocessed = {}
        for source in self.base.expected:
            if source in self.exempt:
                preprocessed[source] = inputs[source]
            else:
                preprocessed[source] = nd.drop(inputs[source], func=self._match)

        return MappingProxyType(preprocessed)

    def _match(self, address):

        return address[-1].ctype in self.accept
