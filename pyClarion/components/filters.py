"""Tools for filtering inputs and outputs of propagators."""


__all__ = ["Gated", "Filtered", "FilteringRelay"]


from ..base.symbols import (
    Symbol, ConstructType, feature, subsystem, terminus
)
from ..base import numdicts as nd
from ..base.components import FeatureInterface, Propagator
from ..utils.funcs import group_by_dims, collect_cmd_data

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


class FilteringRelay(Propagator):
    """Computes gate and filter settings as directed by a controller."""
    
    _serves = ConstructType.buffer

    @dataclass
    class Interface(FeatureInterface):
        """
        Control features for filtering relay.
        
        Defines mapping for assignment of filter weights to cilent constructs 
        based on controller instructions.

        Warning: Do not mutate attributes after creation. Changes will not be 
        reflected.

        :param mapping: Mapping from controller dimension tags to either 
            symbols naming individual clients or a set of symbols for a 
            group of clients. 
        :param vals: A tuple defining feature values corresponding to each 
            strength degree. The i-th value is taken to correspond to a 
            filter weighting level of i / (len(vals) - 1).
        """

        mapping: Mapping[Hashable, Union[Symbol, Set[Symbol]]]
        vals: Tuple[Hashable, ...]

        def _set_interface_properties(self) -> None:

            tv_pairs = product(self.mapping, self.vals)
            cmd_list = list(feature(tag, val) for tag, val in tv_pairs)
            default = self.vals[0]
            default_set = set(feature(tag, default) for tag in self.mapping)

            self._cmds = frozenset(cmd_list)
            self._defaults = frozenset(default_set)
            self._params = frozenset()

        def _validate_data(self):

            if len(set(self.vals)) < 2:
                msg = "Arg `vals` must define at least 2 unique values."
                raise ValueError(msg)

    def __init__(
        self,
        controller: Tuple[subsystem, terminus],
        interface: Interface
    ) -> None:

        self.controller = controller
        self.interface = interface

    def expects(self, construct):

        return construct == self.controller[0]

    def call(self, inputs):

        # This needs badly to be updated to support parameters and take full 
        # advantage of numdicts. - Can

        data = collect_cmd_data(self.client, inputs, self.controller)
        cmds = self.interface.parse_commands(data)

        d = {}
        for dim in self.interface.cmd_dims:

            cmd = cmds[dim]
            tag, _ = dim  

            i, n = self.interface.vals.index(cmd), len(self.interface.vals)
            strength = i / (n - 1)

            entry = self.interface.mapping[tag]
            if not isinstance(entry, Symbol): # entry of type Set[Symbol, ...]
                for client in entry:
                    d[client] = strength
            else: # entry of type Symbol
                d[entry] = strength

        return nd.NumDict(d)
