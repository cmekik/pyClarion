"""Tools for filtering inputs and outputs of propagators."""


__all__ = ["GatedA", "FilteredT", "FilteringRelay"]


from pyClarion.base.symbols import (
    Symbol, ConstructType, feature, subsystem, terminus
)
from pyClarion.base.components import FeatureInterface
from pyClarion.components.propagators import (
    PropagatorA, PropagatorB, PropagatorT
)
from pyClarion.utils.funcs import (
    scale_strengths, multiplicative_filter, group_by_dims, invert_strengths, 
    eye, inv, collect_cmd_data
)

from itertools import product
from dataclasses import dataclass
from typing import NamedTuple, Tuple, Hashable, Union, Mapping, Set, Iterable
from types import MappingProxyType
import pprint


class GatedA(PropagatorA):
    """Gates output of an activation propagator."""
    
    tfms = {"eye": eye, "inv": inv}

    def __init__(
        self, 
        base: PropagatorA, 
        gate: Symbol,
        tfm: str = "eye"
    ) -> None:

        self.base = base
        self.gate = gate
        self.tfm = self.tfms[tfm]

    def expects(self, construct):

        return construct == self.gate or self.base.expects(construct)

    def call(self, construct, inputs):

        weight = inputs[self.gate][construct]
        items = inputs.items()
        _inputs = {src: data for src, data in items if self.base.expects(src)}
        base_strengths = self.base.call(construct, MappingProxyType(_inputs))
        output = scale_strengths(
            weight=self.tfm(weight), 
            strengths=base_strengths
        )

        return output


class FilteredT(PropagatorT):
    """Filters input to a terminus."""
    
    def __init__(
        self, 
        base: PropagatorT, 
        filter: Symbol, 
        invert_weights: bool = True
    ) -> None:

        self.base = base
        self.filter = filter
        self.invert_weights = invert_weights

    def expects(self, construct):

        return construct == self.filter or self.base.expects(construct)

    def call(self, construct, inputs):

        weights = inputs[self.filter]
        
        if self.invert_weights:
            weights = invert_strengths(weights)
            fdefault=1.0
        else:
            fdefault=0.0

        filtered_inputs = {
            src: multiplicative_filter(weights, strengths, fdefault)
            for src, strengths in inputs.items() if self.base.expects(src)
        }
        output = self.base.call(construct, MappingProxyType(filtered_inputs))

        return output


class FilteringRelay(PropagatorB):
    """Computes gate and filter settings as directed by a controller."""
    
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
            feature_list = list(feature(tag, val) for tag, val in tv_pairs)
            default = self.vals[0]
            default_set = set(feature(tag, default) for tag in self.mapping)

            self._features = frozenset(feature_list)
            self._defaults = frozenset(default_set)
            self._tags = frozenset(f.tag for f in self._features)
            self._dims = frozenset(f.dim for f in self._features)

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

    def call(self, construct, inputs):

        data = collect_cmd_data(construct, inputs, self.controller)
        cmds = self.interface.parse_commands(data)

        d = {}
        for dim in self.interface.dims:

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

        return d
