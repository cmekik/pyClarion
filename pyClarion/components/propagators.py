"""Provides some basic propagators for building pyClarion agents."""


__all__ = [
    "PropagatorA", "PropagatorT", "PropagatorB",
    "MaxNodes", "Repeater", "Lag", "ThresholdSelector", "ActionSelector", 
    "BoltzmannSelector", "ConstantBuffer", "Stimulus"
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


####################
### Abstractions ###
####################


class PropagatorA(Propagator):
    """
    Propagator for flows and other activation propagators.

    Maps activations to activations.
    """

    @staticmethod
    def emit(data: Dict[Symbol, float] = None
    ) -> Mapping[Symbol, float]:

        data = data if data is not None else dict()
        return MappingProxyType(mapping=data)


class PropagatorT(Propagator):
    """
    Propagator for subsystem termini.

    Maps activations to decisions.
    """

    _serves = ConstructType.terminus

    @staticmethod
    def emit(data: Set[Symbol] = None) -> FrozenSet[Symbol]:
        
        selection = data or set()
        return frozenset(selection)


class PropagatorB(Propagator):
    """
    Propagator for buffers.

    Maps subsystem outputs to activations.
    """

    _serves = ConstructType.buffer
    
    @staticmethod
    def emit(data: Dict[Symbol, float] = None) -> Mapping[Symbol, float]:

        data = data if data is not None else dict()
        return MappingProxyType(mapping=data)


########################
### Node Propagators ###
########################


class MaxNodes(PropagatorA):

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
                    d[node] = max(d.get(node, 0.0), s)

        return d


########################
### Flow Propagators ###
########################


class Repeater(PropagatorA):
    """Copies the output of a single source construct."""

    _serves = ConstructType.flow_in

    def __init__(self, source: Symbol) -> None:

        self.source = source

    def expects(self, construct):

        return construct == self.source

    def call(self, inputs):

        return dict(inputs[self.source])


class Lag(PropagatorA):
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
            for f, s in strengths.items() if f.lag < self.max_lag
        }

        return d


############################
### Terminus Propagators ###
############################


class ThresholdSelector(PropagatorT):
    """
    Propagator for extracting nodes above a thershold.
    
    Targets feature nodes by default.
    """

    def __init__(self, source: Symbol, threshold: float = 0.85):

        self.source = source
        self.threshold = threshold
        
    def expects(self, construct: Symbol):

        return construct == self.source

    def call(self, inputs):

        strengths = inputs[self.source]
        eligible = (f for f, s in strengths.items() if s > self.threshold)
        return set(eligible)  


class BoltzmannSelector(PropagatorT):
    """Selects a chunk according to a Boltzmann distribution."""

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

        return selection


class ActionSelector(PropagatorT):
    """Selects action paramaters according to Boltzmann distributions."""

    def __init__(self, source, dims, temperature):
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
        self.dims = dims
        self.temperature = temperature

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
        for dim in self.dims:
            # Should there be thresholding here? - CSM
            ipt = {f: s for f, s in strengths.items() if f.dim == dim}
            prs = boltzmann_distribution(ipt, self.temperature)
            sel = select(prs, 1)
            probabilities.update(prs)
            selection.update(sel)

        return selection


##########################
### Buffer Propagators ###
##########################


class ConstantBuffer(PropagatorB):
    """Outputs a stored activation packet."""

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


class Stimulus(PropagatorB):
    """
    Propagates externally provided stimulus.
    
    Offers two possible ways to input stimulus. One possibility is to pass in 
    the 'stimulus' keyword to the client construct at propagation time. The 
    other possibility is to pass the in put in a call to `self.input()`. If 
    both are done the keyword arg dominates.
    """

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
