"""Provides some basic propagators for building pyClarion agents."""


__all__ = [
    "MaxNodes", "Repeater", "Lag", "ThresholdSelector", "ActionSelector", 
    "BoltzmannSelector", "Constants", "Stimulus"
]


from ..base import ConstructType, Symbol, Process, chunk, feature, lag
from .. import numdicts as nd
from .utils import group_by_dims

from typing import (
    Tuple, Mapping, Set, NamedTuple, FrozenSet, Optional, Union, Dict, 
    Sequence, Container
)
from typing import Iterable, Any
from itertools import chain
from copy import copy


########################
### Node Propagators ###
########################


class MaxNodes(Process):
    """Computes the maximum recommended activation for each node in a pool."""

    _serves = ConstructType.nodes
    _ctype_map = {
        ConstructType.features: ConstructType.feature, 
        ConstructType.chunks: ConstructType.chunk
    }

    def __init__(self, sources: Sequence[Symbol]):

        super().__init__(expected=sources)
        self.accept = ConstructType.node

    def entrust(self, path):

        super().entrust(path)
        self.accept = self._ctype_map[self.client[-1].ctype]

    def call(self, inputs):

        data = self.extract_inputs(inputs)
        d = nd.ew_max(*data)
        d = nd.keep(d, func=lambda f: f.ctype in self.accept)
        d = nd.squeeze(d)

        return d


########################
### Flow Propagators ###
########################


class Repeater(Process):
    """Copies the output of a single source construct."""

    _serves = (
        ConstructType.flow_in | ConstructType.flow_h | ConstructType.buffer
    )

    def __init__(self, source: Symbol) -> None:

        super().__init__(expected=(source,))

    def call(self, inputs):

        d, = self.extract_inputs(inputs)
        
        return d


class Lag(Process):
    """Lags strengths for given set of features."""

    _serves = ConstructType.flow_in | ConstructType.flow_bb

    def __init__(self, source: Symbol, max_lag=1):
        """
        Initialize a new `Lag` propagator.

        :param source: Pool of features from which to computed lagged strengths.
        :param max_lag: Do not compute lags beyond this value.
        """

        super().__init__(expected=(source,))
        self.max_lag = max_lag

    def call(self, inputs):

        d, = self.extract_inputs(inputs)
        d = nd.transform_keys(d, func=lag, val=1)
        d = nd.keep(d, func=self._filter)

        return d

    def _filter(self, f):

        return f.ctype in ConstructType.feature and f.lag <= self.max_lag


############################
### Terminus Propagators ###
############################


class ThresholdSelector(Process):
    """
    Propagator for extracting nodes above a thershold.
    
    Targets feature nodes by default.
    """

    _serves = ConstructType.terminus

    def __init__(self, source: Symbol, threshold: float = 0.85):

        super().__init__(expected=(source,))
        self.threshold = threshold
        
    def call(self, inputs):

        d, = self.extract_inputs(inputs)
        d = nd.threshold(d, th=self.threshold, keep_default=True) 

        return d


class BoltzmannSelector(Process):
    """Selects a chunk according to a Boltzmann distribution."""

    _serves = ConstructType.terminus

    def __init__(self, source, temperature=0.01, threshold=0.25):
        """
        Initialize a ``BoltzmannSelector`` instance.

        :param temperature: Temperature of the Boltzmann distribution.
        """

        super().__init__(expected=(source,))
        self.temperature = temperature
        self.threshold = threshold

    def call(self, inputs):
        """
        Select chunks through an activation-based competition. 
        
        Selection probabilities vary with chunk strengths according to a 
        Boltzmann distribution.
        """

        strengths, = self.extract_inputs(inputs)
        thresholded = nd.threshold(strengths, th=self.threshold) 
        probabilities = nd.boltzmann(thresholded, self.temperature)
        d = nd.draw(probabilities, n=1)
        d = nd.with_default(d, default=0)
        
        return d


class ActionSelector(Process):
    """
    Selects actions and paramaters according to Boltzmann distributions.

    Action and parameter features are selected from a given client interface. 
    For parameter features, if a parameter feature is found to be of a 
    singleton dimension (i.e., a dimension with only one value), it is treated 
    like a continuous parameter and its strength is included in the output. If 
    the parameter dimension has multiple values, one among them is 
    stochasticallly selected through a boltzmann distribution, as with action 
    commands.
    """

    _serves = ConstructType.terminus

    def __init__(self, source, interface, temperature):
        """
        Initialize an ``ActionSelector`` instance.

        :param dims: Registered action dimensions.
        :param temperature: Temperature of the Boltzmann distribution.
        """

        if source.ctype not in ConstructType.features:
            raise ValueError("Expected source to be of ctype 'features'.")

        super().__init__(expected=(source,))
        self.interface = interface
        self.temperature = temperature

    def call(self, inputs):
        """
        Select actionable chunks for execution. 
        
        Selection probabilities vary with feature strengths according to a 
        Boltzmann distribution. Probabilities for each target dimension are 
        computed separately.
        """

        strengths, = self.extract_inputs(inputs)
        cmds_by_dims = group_by_dims(self.interface.cmds)
        params_by_dims = group_by_dims(self.interface.params)
        items_by_dims = chain(cmds_by_dims.items(), params_by_dims.items())

        d = nd.MutableNumDict(default=0)
        for dim, fs in items_by_dims:
            if len(fs) == 1: # output strength of singleton param dim
                assert dim in params_by_dims
                f, = fs
                d[f] = strengths[f]
            else: # select value for cmd dim or multivalue param dim
                assert 1 < len(fs)
                ipt = nd.NumDict({f: strengths[f] for f in fs})
                prs = nd.boltzmann(ipt, self.temperature)
                selection = nd.draw(prs, n=1)
                d.update(selection)

        return d


##########################
### Buffer Propagators ###
##########################


class Constants(Process):
    """
    Outputs a constant activation pattern.
    
    Useful for setting defaults and testing. Provides methods for updating 
    constants through external intervention.
    """

    _serves = ConstructType.basic_construct

    def __init__(self, strengths = None) -> None:
        
        self._check_default(strengths)

        super().__init__()
        self.strengths = nd.squeeze(strengths) or nd.NumDict(default=0.0)

    def call(self, inputs):
        """Return stored strengths."""

        return self.strengths

    @staticmethod
    def _check_default(strengths):

        if strengths.default != 0.0:
            msg = "Unexpected default '{}', expected '0'."
            raise ValueError(msg.format(strengths.default))


class Stimulus(Process):
    """Propagates externally provided stimulus."""

    _serves = ConstructType.buffer

    def __init__(self):

        super().__init__()
        self.stimulus = nd.MutableNumDict(default=0.0)

    def input(self, data):

        self.stimulus.update(data)
        self.stimulus.squeeze()

    def call(self, inputs):

        d = self.stimulus
        self.stimulus = nd.MutableNumDict(default=0.0)

        return d
