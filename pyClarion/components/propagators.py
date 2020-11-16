"""Provides some basic propagators for building pyClarion agents."""


__all__ = [
    "MaxNodes", "Repeater", "Lag", "ThresholdSelector", "ActionSelector", 
    "BoltzmannSelector", "Constants", "Stimulus"
]


from ..base import (
    ConstructType, Symbol,  MatchSet, Propagator, chunk, feature,
)
from ..utils.funcs import (
    max_strength, invert_strengths, boltzmann_distribution, select, 
    multiplicative_filter, scale_strengths, linear_rule_strength
)
from typing import (
    Tuple, Mapping, Set, NamedTuple, FrozenSet, Optional, Union, Dict, 
    Sequence, Container
)
from types import MappingProxyType
from collections import namedtuple
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

    def expects(self, construct):

        return construct in self.sources

    def call(self, inputs):

        d = {}
        expected_ctype = self._ctype_map[self.client.ctype]
        for strengths in inputs.values():
            for node, s in strengths.items():
                if node.ctype in expected_ctype:
                    strength = max(d.get(node, 0.0), s)
                    if strength > 0.0: 
                        d[node] = strength

        return d


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

    def expects(self, construct):

        return construct == self.source

    def call(self, inputs):

        return {n: s for n, s in inputs[self.source].items() if s > 0.0}


class Lag(Propagator):
    """Lags strengths for given set of features."""

    _serves = ConstructType.flow_in | ConstructType.flow_bb

    def __init__(self, source: Symbol, max_lag=1):
        """
        Initialize a new `Lag` propagator.

        :param source: Pool of features from which to computed lagged strengths.
        :param max_lag: Do not compute lags beyond this value.
        """

        if source.ctype not in ConstructType.features:
            raise ValueError("Expected construct type to be 'features'.")

        self.source = source
        self.max_lag = max_lag

    def expects(self, construct: Symbol):

        return construct == self.source

    def call(self, inputs):

        strengths = inputs[self.source]
        d = {
            feature(f.tag, f.val, f.lag + 1): s 
            for f, s in strengths.items() 
            if f.ctype in ConstructType.feature and f.lag < self.max_lag
        }

        return d


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
        
    def expects(self, construct: Symbol):

        return construct == self.source

    def call(self, inputs):

        strengths = inputs[self.source]
        eligible = (f for f, s in strengths.items() if s > self.threshold)
        return {n: 1.0 for n in eligible}  


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

    def expects(self, construct: Symbol):

        return construct == self.source

    def call(self, inputs):
        """Select actionable chunks for execution. 
        
        Selection probabilities vary with chunk strengths according to a 
        Boltzmann distribution.
        """

        raw_strengths = inputs[self.source].items()
        strengths = {n: s for n, s in raw_strengths if self.threshold < s}
        probabilities = boltzmann_distribution(strengths, self.temperature)
        selection = select(probabilities, 1)

        return {n: 1.0 for n in selection}


class ActionSelector(Propagator):
    """Selects action paramaters according to Boltzmann distributions."""

    _serves = ConstructType.terminus

    def __init__(self, source, client_interface, temperature, default=0.0):
        """
        Initialize a ``ActionSelector`` instance.

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
        self.default = default

    def expects(self, construct):
        
        return construct == self.source 

    def call(self, inputs):
        """Select actionable chunks for execution. 
        
        Selection probabilities vary with feature strengths according to a 
        Boltzmann distribution. Probabilities for each target dimension are 
        computed separately.
        """

        strengths = inputs[self.source]
        probabilities, selection = dict(), set()
        for dim, fs in self.client_interface.features_by_dims.items():
            ipt = {f: strengths.get(f, self.default) for f in fs}
            prs = boltzmann_distribution(ipt, self.temperature)
            sel = select(prs, 1)
            probabilities.update(prs)
            selection.update(sel)

        return {n: 1.0 for n in selection}


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

        self.strengths = strengths or dict()

    def expects(self, construct: Symbol):

        return False

    def call(self, inputs):
        """Return stored strengths."""

        return self.strengths

    def update_strengths(self, strengths):
        """Update self with contents of dict-like strengths."""

        self.strengths = self.strengths.copy()
        self.strengths.update(strengths)

    def clear(self) -> None:
        """Clear stored node strengths."""

        self.strengths = {}


class Stimulus(Propagator):
    """Propagates externally provided stimulus."""

    _serves = ConstructType.buffer

    def __init__(self):

        self.stimulus = {}

    def expects(self, construct: Symbol):

        return False

    def input(self, data):

        self.stimulus.update(data)

    def call(self, inputs, stimulus=None):

        d = stimulus if stimulus is not None else self.stimulus
        self.stimulus = {}

        return d
