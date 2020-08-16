"""Provides some basic propagators for building pyClarion agents."""


__all__ = [
    "PropagatorN", "PropagatorA", "PropagatorT", "PropagatorB",
    "MaxNode", "Lag", "ThresholdSelector", "ActionSelector", 
    "BoltzmannSelector", "ConstantBuffer", "Stimulus", "FilteredN", "FilteredT"
]


from pyClarion.base import (
    ConstructType, Symbol,  MatchSet, Propagator, chunk, feature,
)
from pyClarion.utils.funcs import (
    max_strength, boltzmann_distribution, select, multiplicative_filter, 
    scale_strengths, linear_rule_strength
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
    Propagator for a collection of nodes or a single flows.

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


class Lag(PropagatorA):
    """Lags strengths for given features."""

    # Expected dimension type. Lag object looks for a `lag` attribute, which 
    # is an int specifying lag amount. Any symbolic object satisfying this 
    # requirement may be used.
    Dim = namedtuple("LagDim", ["name", "lag"])

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
            feature(
                dim=type(self).Dim(name=f.dim.name, lag=f.dim.lag + 1), 
                val=f.val
            ): s 
            for f, s in inputs.items() if f.dim.lag < self.max_lag
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
    """Propagates externally provided stimulus."""

    def call(self, construct, inputs, stimulus=None, **kwds):

        return stimulus or {}


##########################
### Filtering Wrappers ###
##########################

# WARNING: Filtering may be broken after changes to propagators... - Can


class FilteredN(PropagatorN):
    """Filters input and output activations of an activation propagator."""
    
    def __init__(
        self, 
        base: PropagatorN, 
        source_filter: Symbol = None, 
        input_filter: Symbol = None, 
        output_filter: Symbol = None, 
        fdefault=0.0,
    ):

        self.base = base
        # Expected types for source_filter, input_filter and output_filter 
        # should be construct symbols.
        self.source_filter = source_filter
        self.input_filter = input_filter
        self.output_filter = output_filter
        self.fdefault = fdefault

    def __copy__(self):

        return type(self)(
            base=copy(self.base),
            source_filter=copy(self.source_filter),
            input_filter=copy(self.input_filter),
            output_filter=copy(self.output_filter),
            fdefault=copy(self.fdefault)
        )

    def expects(self, construct):

        b = False
        for c in (self.source_filter, self.input_filter, self.output_filter):
            if c is not None:
                b |= construct == c 

        return b or self.base.expects(construct=construct)

    def call(self, construct, inputs, **kwds):

        # Get filter settings and remove filter info from inputs dict so they 
        # are not processed by self.base.
        if self.source_filter is not None:
            source_weights = inputs.pop(self.source_filter)
        if self.input_filter is not None:
            input_weights = inputs.pop(self.input_filter)
        if self.output_filter is not None:
            output_weights = inputs.pop(self.output_filter)

        # Apply source filtering
        if self.source_filter is not None:
            inputs = {
                source: scale_strengths(
                    weight=source_weights.get(source, 1.0 - self.fdefault), 
                    strengths=packet, 
                ) 
                for source, packet in inputs.items()
            }

        # Filter inputs to base
        if self.input_filter is not None:
            inputs = multiplicative_filter(
                filter_weights=input_weights, 
                strengths=inputs, 
                fdefault=self.fdefault
            )
        
        # Call base on (potentially) filtered inputs. Note that call is to 
        # `base.call()` instead of `base.__call__()`. This is because we rely 
        # on `self.__call__()` instead.
        output = self.base.call(construct, inputs, **kwds)

        # Filter outputs of base
        if self.output_filter is not None:
            output = multiplicative_filter(
                filter_weights=output_weights, 
                strengths=output, 
                fdefault=self.fdefault
            )

        return output


class FilteredT(PropagatorT):
    """Filters input and output activations of a decision propagator."""
    
    def __init__(
        self, 
        base: PropagatorT, 
        source_filter: Symbol = None,
        input_filter: Symbol = None, 
        fdefault=0.0
    ):

        self.base = base
        self.source_filter = source_filter
        self.input_filter = input_filter
        self.fdefault = fdefault

    def expects(self, construct):

        b = False
        for c in (self.source_filter, self.input_filter):
            if c is not None:
                b |= construct == c 

        return b or self.base.expects(construct=construct)

    def call(self, construct, inputs, **kwds):

        # Get filter settings and remove filter info from inputs dict so they 
        # are not processed by self.base
        if self.source_filter is not None:
            source_weights = inputs.pop(self.source_filter)
        if self.input_filter is not None:
            input_weights = inputs.pop(self.input_filter)

        # Apply source filtering
        if self.source_filter is not None:
            inputs = {
                source: scale_strengths(
                    weight=source_weights.get(source, 1.0 - self.fdefault), 
                    strengths=packet, 
                ) 
                for source, packet in inputs.items()
            }

        # Filter inputs to base
        if self.input_filter is not None:
            inputs = multiplicative_filter(
                filter_weights=input_weights, 
                strengths=inputs, 
                fdefault=self.fdefault
            )
        
        # Call base on (potentially) filtered inputs. Note that call is to 
        # `base.call()` instead of `base.__call__()`. This is because we rely 
        # on `self.__call__()` instead.
        output = self.base.call(construct, inputs, **kwds)

        return output
