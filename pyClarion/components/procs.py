"""Provides basic propagators for building pyClarion agents."""


__all__ = [
    "MaxNode", "BoltzmannSelector", "ConstantBuffer", "Stimulus", 
    "FilteredA", "FilteredD"
]


from pyClarion.base import ConstructSymbol, ActivationPacket, DecisionPacket
from pyClarion.base.propagators import PropagatorA, PropagatorB, PropagatorD
from pyClarion.utils.funcs import (
    max_strength, simple_junction, boltzmann_distribution, select, 
    multiplicative_filter, scale_strengths
)


class MaxNode(PropagatorA):
    """Simple node returning maximum strength for given construct."""

    def call(self, construct, inputs, **kwds):

        packets = inputs.values()
        strength = max_strength(construct, packets)
        
        return strength


class BoltzmannSelector(PropagatorD):
    """Selects a chunk according to a Boltzmann distribution."""

    def __init__(self, temperature, k=1):
        """
        Initialize a ``BoltzmannSelector`` instance.

        :param temperature: Temperature of the Boltzmann distribution.
        """

        self.temperature = temperature
        self.k = k

    def call(self, construct, inputs, **kwds):
        """Select actionable chunks for execution. 
        
        Selection probabilities vary with chunk strengths according to a 
        Boltzmann distribution.

        :param strengths: Mapping of node strengths.
        """

        packets = inputs.values()
        strengths = simple_junction(packets)
        probabilities = boltzmann_distribution(strengths, self.temperature)
        selection = select(probabilities, self.k)

        return probabilities, selection


class ConstantBuffer(PropagatorB):
    """Outputs a stored activation packet."""

    def __init__(self, strengths = None) -> None:

        self.strengths = strengths or dict()

    def call(self, construct, inputs, **kwds):
        """Return stored strengths."""

        return self.strengths

    def update(self, strengths):
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


class FilteredA(PropagatorA):
    """Filters input and output activations of an activation propagator."""
    
    def __init__(
        self, 
        base: PropagatorA, 
        source_filter: ConstructSymbol = None, 
        input_filter: ConstructSymbol = None, 
        output_filter: ConstructSymbol = None, 
        fdefault=0
    ):

        self.base = base
        # Expected types for source_filter, input_filter and output_filter 
        # should be construct symbols.
        self.source_filter = source_filter
        self.input_filter = input_filter
        self.output_filter = output_filter
        self.fdefault = fdefault

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
                    weight=source_weights.get(source, 1 - self.fdefault), 
                    strengths=packet, 
                ) 
                for source, packet in inputs.items()
            }

        # Filter inputs to base
        if self.input_filter is not None:
            inputs = {
                source: multiplicative_filter(
                    filter_weights=input_weights, 
                    strengths=packet, 
                    fdefault=self.fdefault
                )
                for source, packet in inputs.items()
            }
        
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


class FilteredD(PropagatorD):
    """Filters input and output activations of a decision propagator."""
    
    def __init__(
        self, 
        base: PropagatorD, 
        input_filter: ConstructSymbol = None, 
        fdefault=0
    ):

        self.base = base
        self.input_filter = input_filter
        self.fdefault = fdefault

    def call(self, construct, inputs, **kwds):

        # Get filter settings and remove filter info from inputs dict so they 
        # are not processed by self.base
        if self.input_filter is not None:
            input_weights = inputs.pop(self.input_filter)

        # Filter inputs to base
        if self.input_filter is not None:
            inputs = {
                source: multiplicative_filter(
                    filter_weights=input_weights, 
                    strengths=packet, 
                    fdefault=self.fdefault
                )
                for source, packet in inputs.items()
            }
        
        # Call base on (potentially) filtered inputs. Note that call is to 
        # `base.call()` instead of `base.__call__()`. This is because we rely 
        # on `self.__call__()` instead.
        output = self.base.call(construct, inputs, **kwds)

        return output
