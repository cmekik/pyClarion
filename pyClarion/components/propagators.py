"""Provides some basic propagators for building pyClarion agents."""


__all__ = [
    "MaxNodes", "Repeater", "Lag", "ThresholdSelector", "ActionSelector", 
    "BoltzmannSelector", "Constants", "Stimulus"
]


from ..base import ConstructType, Symbol, Propagator, chunk, feature, lag
from .. import numdicts as nd

from typing import (
    Tuple, Mapping, Set, NamedTuple, FrozenSet, Optional, Union, Dict, 
    Sequence, Container
)
from typing import Iterable, Any
from copy import copy


########################
### Node Propagators ###
########################


class MaxNodes(Propagator):
    """Computes the maximum recommended activation for each node in a pool."""

    _serves = ConstructType.nodes
    _ctype_map = {
        ConstructType.features: ConstructType.feature, 
        ConstructType.chunks: ConstructType.chunk
    }

    def __init__(self, sources: Container[Symbol]):

        self.sources = sources

    @property
    def expected(self):

        return frozenset(self.sources)

    def call(self, inputs):

        d = nd.ew_max(*inputs.values())

        return nd.keep(d, func=self._filter)

    def _filter(self, f):

        return f.ctype in self._ctype_map[self.client.ctype]


########################
### Flow Propagators ###
########################


class Repeater(Propagator):
    """Copies the output of a single source construct."""

    _serves = (
        ConstructType.flow_in | ConstructType.flow_h | ConstructType.buffer
    )

    def __init__(self, source: Symbol) -> None:

        self.source = source

    @property
    def expected(self):

        return frozenset((self.source,))

    def call(self, inputs):

        return inputs[self.source]


class Lag(Propagator):
    """Lags strengths for given set of features."""

    _serves = ConstructType.flow_in | ConstructType.flow_bb

    def __init__(self, source: Symbol, max_lag=1):
        """
        Initialize a new `Lag` propagator.

        :param source: Pool of features from which to computed lagged strengths.
        :param max_lag: Do not compute lags beyond this value.
        """

        self.source = source
        self.max_lag = max_lag

    @property
    def expected(self):

        return frozenset((self.source,))

    def call(self, inputs):

        d = nd.transform_keys(inputs[self.source], func=lag, val=1)
        d = nd.keep(d, func=self._filter)

        return d

    def _filter(self, f):

        return f.ctype in ConstructType.feature and f.lag <= self.max_lag


############################
### Terminus Propagators ###
############################


class ThresholdSelector(Propagator):
    """
    Propagator for extracting nodes above a thershold.
    
    Targets feature nodes by default.
    """

    _serves = ConstructType.terminus

    def __init__(self, source: Symbol, threshold: float = 0.85):

        self.source = source
        self.threshold = threshold
        
    @property
    def expected(self):

        return frozenset((self.source,))

    def call(self, inputs):

        return nd.threshold(inputs[self.source], self.threshold) 


class BoltzmannSelector(Propagator):
    """Selects a chunk according to a Boltzmann distribution."""

    _serves = ConstructType.terminus

    def __init__(self, source, temperature=0.01, threshold=0.25):
        """
        Initialize a ``BoltzmannSelector`` instance.

        :param temperature: Temperature of the Boltzmann distribution.
        """

        self.source = source
        self.temperature = temperature
        self.threshold = threshold

    @property
    def expected(self):

        return frozenset((self.source,))

    def call(self, inputs):
        """Select actionable chunks for execution. 
        
        Selection probabilities vary with chunk strengths according to a 
        Boltzmann distribution.
        """

        strengths = inputs[self.source]
        thresholded = nd.threshold(strengths, th=self.threshold) 
        probabilities = nd.boltzmann(thresholded, self.temperature)
        d = nd.draw(probabilities, 1)

        return d


class ActionSelector(Propagator):
    """Selects action paramaters according to Boltzmann distributions."""

    _serves = ConstructType.terminus

    def __init__(self, source, client_interface, temperature):
        """
        Initialize an ``ActionSelector`` instance.

        :param dims: Registered action dimensions.
        :param temperature: Temperature of the Boltzmann distribution.
        """

        if source.ctype not in ConstructType.features:
            raise ValueError("Expected source to be of ctype 'features'.")

        # Need to make sure that sparse activation representation doesn't cause 
        # problems. Add self.features to make sure selection is done 
        # consistently? - CSM

        self.source = source
        self.client_interface = client_interface
        self.temperature = temperature

    @property
    def expected(self):

        return frozenset((self.source,))

    def call(self, inputs):
        """Select actionable chunks for execution. 
        
        Selection probabilities vary with feature strengths according to a 
        Boltzmann distribution. Probabilities for each target dimension are 
        computed separately.
        """

        strengths = inputs[self.source]
        params = self.client_interface.params
        cmds_by_dims = self.client_interface.cmds_by_dims

        d = nd.MutableNumDict({f: strengths[f] for f in params})

        for dim, fs in cmds_by_dims.items():
            ipt = nd.NumDict({f: strengths[f] for f in fs})
            prs = nd.boltzmann(ipt, self.temperature)
            selection = nd.draw(prs, 1)
            d.update(selection)

        return d


##########################
### Buffer Propagators ###
##########################


class Constants(Propagator):
    """
    Outputs a constant activation pattern.
    
    Useful for setting defaults and testing. Provides methods for updating 
    constants through external intervention.
    """

    _serves = ConstructType.basic_construct

    def __init__(self, strengths = None) -> None:

        self.strengths = strengths or nd.NumDict(default=0.0)

    @property
    def expected(self):

        return frozenset()

    def call(self, inputs):
        """Return stored strengths."""

        return self.strengths

    def update_strengths(self, strengths):
        """Update self with contents of dict-like strengths."""

        self.strengths = strengths

    def clear(self) -> None:
        """Clear stored node strengths."""

        self.strengths = nd.NumDict(default=0.0)


class Stimulus(Propagator):
    """Propagates externally provided stimulus."""

    _serves = ConstructType.buffer

    def __init__(self):

        self.stimulus = nd.MutableNumDict()

    @property
    def expected(self):

        return frozenset()

    def input(self, data):

        self.stimulus.update(data)

    def call(self, inputs):

        d = self.stimulus
        self.stimulus = nd.MutableNumDict()

        return d
