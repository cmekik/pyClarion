"""Tools for filtering inputs and outputs of propagators."""


__all__ = ["Gated", "Filtered", "Pruned"]


from ..base.symbols import (
    ConstructType, Symbol, SymbolTrie, SymbolicAddress, 
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

        super().__init__(base=base, expected=(gate,) + base.expected)
        self._expected_top = (gate,)
        self.invert = invert

    def postprocess(
        self, inputs: SymbolTrie[nd.NumDict], output: nd.NumDict
    ) -> nd.NumDict:

        data, = self.extract_inputs(inputs)[:len(self.expected_top)]
        w = data[self.client]
        if self.invert:
            w = 1 - w

        return w * output


def _freeze_inputs(mutable: SymbolTrie[nd.NumDict]) -> SymbolTrie[nd.NumDict]:

    result: SymbolTrie[nd.NumDict] = MappingProxyType(
        {
            k: v if isinstance(v, nd.NumDict) else _freeze_inputs(v) 
            for k, v in mutable.items()
        }
    )

    return result


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

        super().__init__(base=base, expected=(sieve,) + base.expected)
        self._expected_top = (sieve,)
        self.exempt = exempt or set() 
        self.invert = invert

    def preprocess(self, inputs):

        ws, = self.extract_inputs(inputs)[:len(self.expected_top)]
        if self.invert:
            ws = 1 - ws

        preprocessed = {}
        for source in self.base.expected:
            d = preprocessed
            _inputs = inputs
            if isinstance(source, Symbol):
                _source = (source,)
            else:
                _source = source
            for i, key in enumerate(_source):
                if i < len(_source) - 1:
                    d = d.setdefault(key, {})
                    _inputs = inputs[key]
                else:
                    assert i == len(_source) - 1
                    if source in self.exempt:
                        d[key] = _inputs
                    else:
                        d[key] = ws * _inputs[key]

        return _freeze_inputs(preprocessed)


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

        super().__init__(base=base, expected=base.expected)
        self._expected_top = ()
        self.accept = accept
        self.exempt = exempt or set() 

    def preprocess(self, inputs):

        preprocessed = {}
        for source in self.base.expected:
            d = preprocessed
            _inputs = inputs
            if isinstance(source, Symbol):
                _source = (source,)
            else:
                _source = source
            for i, key in enumerate(_source):
                if i < len(_source) - 1:
                    d = d.setdefault(key, {})
                    _inputs = inputs[key]
                else:
                    assert i == len(_source) - 1
                    if source in self.exempt:
                        d[key] = _inputs
                    else:
                        d[key] = nd.drop(_inputs[key], func=self._match)

        return _freeze_inputs(preprocessed)

    def _match(self, construct):

        return construct.ctype in self.accept
