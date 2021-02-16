"""Tools for filtering inputs and outputs of propagators."""


__all__ = ["Gated", "Filtered", "Pruned"]


from ..base.symbols import (
    ConstructType, Symbol, SymbolicAddress, 
    feature, buffer, flow_in, subsystem, terminus
)
from ..base.components import Wrapped, Pt
from .. import numdicts as nd
from .buffers import ParamSet

from itertools import product
from dataclasses import dataclass
from typing import (
    NamedTuple, Tuple, Hashable, Union, Mapping, List, Iterable
)
from types import MappingProxyType
import pprint


class Gated(Wrapped[Pt]):
    """
    Gates output of an activation propagator.
    
    The gating function is achieved by multiplying a gating signal associated 
    with the client construct in the output of the gate by the output of the 
    base propagator.

    The gating signal is assumed to be in the interval [0, 1],
    """

    def __init__(
        self, 
        base: Pt, 
        controller: Union[buffer, flow_in],
        interface: ParamSet.Interface, 
        pidx: int,
        invert: bool = False
    ) -> None:
        """
        Initialize a Gated instance.

        :param base: The base Process instance.
        :param controller: The gate controller.
        :param interface: Controller's feature interface.
        :param pidx: Lookup index of gating parameter in interface.
        :param invert: Option to invert the gating signal.
        """

        super().__init__(base=base, expected=(controller,))

        self.pidx = pidx
        self.interface = interface
        self.invert = invert

    def postprocess(self, inputs, output):
        """
        Gate output of base process.

        Multiplies output of base process by gating parameter. If the invert 
        option is selected, will first invert the gating parameter.
        """

        data, = self.extract_inputs(inputs)[:len(self.expected_top)]
        w = data[self.interface.params[self.pidx]]
        if self.invert:
            w = 1 - w

        return w * output


class Filtered(Wrapped[Pt]):
    """
    Filters the input to a propagator.
    
    Filtering is achieved by elementwise multiplication of each input to the 
    base propagator by a filtering signal from a sieve construct. 

    The filtering signal is assumed to be in the interval [0, 1].
    """

    def __init__(
        self, 
        base: Pt, 
        controller: Union[buffer, flow_in],
        exempt: List[SymbolicAddress] = None, 
        invert: bool = True
    ) -> None:

        super().__init__(base=base, expected=(controller,))

        self.exempt = exempt or [] 
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


class Pruned(Wrapped[Pt]):
    """
    Prunes the input to an activation propagator.
    
    Pruning is achieved by removing, in each input to the base propagator, 
    constructs of a chosen construct type. 
    """

    def __init__(
        self, 
        base: Pt, 
        accept: ConstructType,
        exempt: List[SymbolicAddress] = None
    ) -> None:

        super().__init__(base=base)
        self.accept = accept
        self.exempt = exempt or []

    def preprocess(self, inputs):

        preprocessed = {}
        for source in self.base.expected:
            if source in self.exempt:
                preprocessed[source] = inputs[source]
            else:
                preprocessed[source] = nd.keep(
                    d=inputs[source], 
                    # TODO: Fix func: will break if address not tuple. - Can
                    func=lambda symbol: symbol.ctype in self.accept 
                )

        return MappingProxyType(preprocessed)
