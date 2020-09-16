"""Provides some basic propagators for building pyClarion agents."""


__all__ = [
    "PropagatorN", "PropagatorA", "PropagatorT", "PropagatorB",
    "MaxNode", "Repeater", "Lag", "ThresholdSelector", "ActionSelector", 
    "BoltzmannSelector", "ConstantBuffer", "Stimulus"
]


from pyClarion.base import (
    ConstructType, Symbol,  MatchSet, Propagator, chunk, feature,
)
from pyClarion.utils.funcs import (
    max_strength, invert_strengths, boltzmann_distribution, select, 
    multiplicative_filter, scale_strengths, linear_rule_strength
)
from typing import (
    Tuple, Mapping, Set, NamedTuple, FrozenSet, Optional, Union, Dict, Sequence
)
from types import MappingProxyType
from collections import namedtuple
from typing import Iterable, Any
from copy import copy


####################
### Abstractions ###
####################


class PropagatorN(Propagator[Mapping[Symbol, float], float, float]):
    """
    Propagator for individual nodes.

    Maps activations to activations. Default activation is assumed to be zero.
    """

    def emit(self, data: float = None) -> float:

        output = data if data is not None else 0.0
        return output


class PropagatorA(
    Propagator[float, Dict[Symbol, float], Mapping[Symbol, float]]
):
    """
    Propagator for flows and other activation propagators.

    Maps activations to activations.
    """

    def emit(
        self, data: Dict[Symbol, float] = None
    ) -> Mapping[Symbol, float]:

        data = data if data is not None else dict()
        return MappingProxyType(mapping=data)


class PropagatorT(Propagator[float, Set[Symbol], FrozenSet[Symbol]]):
    """
    Propagator for subsystem termini.

    Maps activations to decisions.
    """

    def emit(self, data: Set[Symbol] = None) -> FrozenSet[Symbol]:
        
        selection = data or set()
        return frozenset(selection)


class PropagatorB(
    Propagator[
        Mapping[Symbol, Any], Dict[Symbol, float], Mapping[Symbol, float]
    ]
):
    """
    Propagator for buffers.

    Maps subsystem outputs to activations.
    """
    
    def emit(self, data: Dict[Symbol, float] = None) -> Mapping[Symbol, float]:

        data = data if data is not None else dict()
        return MappingProxyType(mapping=data)

    def update(self, construct, inputs):
        """
        Update buffer state based on inputs.
        
        :param construct: Name of the client construct. 
        :param inputs: Pairs the names of input constructs with their outputs.
        """

        pass

    class StateUpdater(object):
        """
        Updater for PropagatorB instances.

        Delegates to propagator's `update()` method.
        """

        def __call__(self, realizer):

            # Add a check here to make sure that the updater behaves as 
            # intended? Seems difficult to implement correctly. - Can

            _inputs = realizer.inputs
            
            construct = realizer.construct
            inputs = {src: pull_func() for src, pull_func in _inputs.items()}

            realizer.emitter.update(construct, inputs)


########################
### Node Propagators ###
########################


class MaxNode(PropagatorN):
    """Simple node returning maximum strength for given construct."""

    def __copy__(self):

        return type(self)(matches=copy(self.matches))

    def call(self, construct, inputs, **kwds):

        packets = inputs.values()
        strength = max_strength(construct, packets)
        
        return strength


########################
### Flow Propagators ###
########################


class Repeater(PropagatorA):
    """Copies the output of a single source construct."""

    def __init__(self, source: Symbol) -> None:

        super().__init__()
        self.source = source

    def expects(self, construct):

        return construct == self.source

    def call(self, construct, inputs, **kwds):

        return dict(inputs[self.source])


class Lag(PropagatorA):
    """Lags strengths for given features."""

    def __init__(self, max_lag=1, matches=None):
        """
        Initialize a new `Lag` propagator.

        :param max_lag: Do not compute lags beyond this value.
        """

        if matches is None: 
            matches = MatchSet(ctype=ConstructType.feature)  
        super().__init__(matches=matches)

        self.max_lag = max_lag

    def call(self, construct, inputs, **kwds):

        d = {
            feature(f.dlb, f.val, f.lag + 1): s 
            for f, s in inputs.items() if f.lag < self.max_lag
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

    def __init__(self, threshold=0.85, matches=None):

        if matches is None: 
            matches = MatchSet(ctype=ConstructType.feature)  
        super().__init__(matches=matches)
        self.threshold = threshold
        
    def call(self, construct, inputs, **kwds):

        eligible = (f for f, s in inputs.items() if s > self.threshold)
        return set(eligible)  


class BoltzmannSelector(PropagatorT):
    """Selects a chunk according to a Boltzmann distribution."""

    def __init__(self, temperature, threshold=0.25, matches=None):
        """
        Initialize a ``BoltzmannSelector`` instance.

        :param temperature: Temperature of the Boltzmann distribution.
        """

        super().__init__(matches=matches)

        self.temperature = temperature
        self.threshold = threshold

    def call(self, construct, inputs, **kwds):
        """Select actionable chunks for execution. 
        
        Selection probabilities vary with chunk strengths according to a 
        Boltzmann distribution.

        :param strengths: Mapping of node strengths.
        """

        inputs = {n: s for n, s in inputs.items() if self.threshold < s}
        probabilities = boltzmann_distribution(inputs, self.temperature)
        selection = select(probabilities, 1)

        return selection


class ActionSelector(PropagatorT):
    """Selects action paramaters according to Boltzmann distributions."""

    def __init__(self, dims, temperature):
        """
        Initialize a ``ActionSelector`` instance.

        :param dims: Registered action dimensions.
        :param temperature: Temperature of the Boltzmann distribution.
        """

        self.dims = dims
        self.temperature = temperature

    def expects(self, construct):
        
        return (
            construct.ctype in ConstructType.feature and 
            construct.dim in self.dims
        )

    def call(self, construct, inputs, **kwds):
        """Select actionable chunks for execution. 
        
        Selection probabilities vary with feature strengths according to a 
        Boltzmann distribution. Probabilities for each target dimension are 
        computed separately.
        """

        probabilities, selection = dict(), set()
        for dim in self.dims:
            ipt = {f: s for f, s in inputs.items() if f.dim == dim}
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

    def __init__(self, strengths = None, matches = None) -> None:

        super().__init__(matches=matches)
        self.strengths = strengths or dict()

    def call(self, construct, inputs, **kwds):
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

        super().__init__()
        self.stimulus = {}

    def input(self, data):

        self.stimulus.update(data)

    def call(self, construct, inputs, stimulus=None, **kwds):


        d = stimulus if stimulus is not None else self.stimulus
        self.stimulus = {}

        return d
