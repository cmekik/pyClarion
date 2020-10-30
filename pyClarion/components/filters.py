"""Tools for filtering inputs and outputs of propagators."""


__all__ = ["GatedA", "FilteredT", "FilteringRelay"]


from pyClarion.base.symbols import Symbol, ConstructType, feature
from pyClarion.base.realizers import FeatureInterface
from pyClarion.components.propagators import (
    PropagatorA, PropagatorB, PropagatorT
)
from pyClarion.utils.funcs import (
    scale_strengths, multiplicative_filter, group_by_dims, invert_strengths, 
    eye, inv
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

        weight = inputs.pop(self.gate)[construct]
        base_strengths = self.base.call(construct, inputs)
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

        weights = inputs.pop(self.filter)
        
        if self.invert_weights:
            weights = invert_strengths(weights)
            fdefault=1.0
        else:
            fdefault=0.0

        filtered_inputs = {
            source: multiplicative_filter(
                weights=weights, strengths=strengths, fdefault=fdefault
                )
            for source, strengths in inputs.items()
        }
        output = self.base.call(construct, filtered_inputs)

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

        def __post_init__(self):

            self._validate_data(self.mapping, self.vals)
            self._set_interface_properties()

        def _set_interface_properties(self) -> None:

            tv_pairs = product(self.mapping, self.vals)
            feature_list = list(feature(tag, val) for tag, val in tv_pairs)
            default = self.vals[0]
            default_set = set(feature(tag, default) for tag in self.mapping)
            default_dict = {f.dim: f for f in default_set}

            self._features = frozenset(feature_list)
            self._defaults = MappingProxyType(default_dict)
            self._tags = frozenset(f.tag for f in self._features)
            self._dims = frozenset(f.dim for f in self._features)

        def _validate_data(self, mapping, vals):

            if len(set(vals)) < 2:
                msg = "Arg `vals` must define at least 2 unique values."
                raise ValueError(msg)

    def __init__(
        self,
        controller: Tuple[Symbol, Symbol],
        interface: Interface
    ) -> None:

        self._validate_controller(controller)

        super().__init__()
        self.controller = controller
        self.interface = interface

    def expects(self, construct):

        return construct == self.controller[0]

    def call(self, construct, inputs):
        
        d, cmds = {}, self._parse_commands(inputs)
        for dim in self.interface.dims:
            tag, lag = dim
            cmd = cmds.get(dim, self.interface.defaults[dim])
            level = self.interface.vals.index(cmd.val)
            strength = level / (len(self.interface.vals) - 1)
            entry = self.interface.mapping[tag]
            if not isinstance(entry, Symbol): # entry of type Set[Symbol, ...]
                for client in entry:
                    d[client] = strength
            else: # entry of type Symbol
                d[entry] = strength
        return d

    def _parse_commands(self, inputs):

        subsystem, terminus = self.controller
        data = inputs[subsystem].get(terminus, frozenset())

        # Filter irrelevant feature symbols
        cmd_set = set(
            f for f in data if 
            f in self.interface.features and 
            f.tag in self.interface.tags and
            f.lag == 0
        )

        groups = group_by_dims(features=cmd_set)
        cmds = {}
        for k, g in groups.items():
            if len(g) > 1:
                raise ValueError(
                "Multiple commands for dim '{}' in FilterBus.".format(k)
            )
            cmds[k] = g[0]

        return cmds

    @staticmethod
    def _validate_controller(controller):

        subsystem, terminus = controller
        if subsystem.ctype not in ConstructType.subsystem:
            raise ValueError(
                "Arg `controller` must name a subsystem at index 0."
            )
        if terminus.ctype not in ConstructType.terminus:
            raise ValueError(
                "Arg `controller` must name a terminus at index 1."
            )

